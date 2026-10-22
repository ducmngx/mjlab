"""MPPI over the impact model.

One cycle (every ``step_s``) for ``n`` robots:

1. sample ``k`` plans per robot (``PlanSampler``), deadbanded as the
   command term will apply them, and integrate their commanded paths;
2. predict drift (and its spread), falls and trunk contact for every plan
   with the dynamics model (one encode per robot);
3. cost each plan: progress to the goal, the footprint sweep of the
   predicted path ``cmd_path + drift +- k sigma`` over the obstacle map,
   unknown cells, P(fallen by the end), P(trunk contact), smoothness, and on
   the course the offset from the lane centre;
4. weight plans by ``exp(-(cost - min) / (temperature * (median - min)))``
   and average them into the new plan, kept (shifted) as the next warm
   start. Scaling by the median gap, not the spread, keeps a few hopeless
   plans (stops, holds into a wall) from flattening the weights.

With ``delay_steps`` the first steps of every plan are the commands already
committed: on the robot, a plan computed from the state at ``t`` starts
executing one cycle later, and the sim runs the same way.

Inputs are in each robot's current yaw frame (origin at the trunk, +x
forward): the goal, the lane, and the map, which shares the frame of the
plans' predicted paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from failbench.mppi.costmap import FootprintCfg, distance_field, obstacle_cells, sweep
from failbench.mppi.inputs import deadband, integrate_plan
from failbench.mppi.models import DynamicsModel
from failbench.mppi.sampler import CRUISE_VX, PlanSampler, SamplerCfg


@dataclass(frozen=True)
class CostWeights:
  goal: float = 1.0
  """Per metre from the goal at the plan's closest approach (passing it is
  fine: the robot stops on arrival)."""
  goal_running: float = 0.5
  """Per metre from the goal, averaged over the plan."""
  lane: float = 0.5
  """Per m^2 of lateral offset from the lane centre, averaged."""
  obstacle: float = 20.0
  """Per metre of footprint penetration, summed over steps and circles."""
  unknown: float = 0.2
  """Per footprint circle on an unknown cell, averaged over steps."""
  fall: float = 20.0
  collision: float = 5.0
  smooth: float = 0.5
  """Per (m/s)^2 or (rad/s)^2 of change between plan steps, summed."""


@dataclass(frozen=True)
class MppiCfg:
  sampler: SamplerCfg = field(default_factory=SamplerCfg)
  footprint: FootprintCfg = field(default_factory=FootprintCfg)
  weights: CostWeights = field(default_factory=CostWeights)
  temperature: float = 0.25
  control_dt: float = 0.02
  """Control step the plans are held over (HIM runs at 50 Hz)."""
  delay_steps: int = 1
  zero_below: float = 0.2
  """The command term's deadband (``GoalVelocityCommandCfg.zero_below``)."""


class Mppi:
  def __init__(self, cfg: MppiCfg, model: DynamicsModel, n: int, device):
    self.cfg = cfg
    self.model = model
    self.sampler = PlanSampler(cfg.sampler, device)
    s = cfg.sampler
    self.substeps = round(s.step_s / cfg.control_dt)
    self.plan = torch.zeros(n, s.steps, 3, device=device)
    """Each robot's current plan; ``plan[:, 0]`` runs now."""

  def reset(self, ids: torch.Tensor, cmd: torch.Tensor | None = None) -> None:
    self.plan[ids] = 0.0 if cmd is None else cmd[:, None]

  def advance(self) -> torch.Tensor:
    """Move one step along the current plans without replanning (a planner
    that commits to a plan for several cycles); returns the commands to run
    now."""
    self.plan = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], dim=1)
    return self.plan[:, 0]

  def _path(self, plans: torch.Tensor) -> torch.Tensor:
    return integrate_plan(plans, self.cfg.control_dt, self.substeps)

  @torch.no_grad()
  def score(
    self,
    inputs: dict[str, torch.Tensor],
    plans: torch.Tensor,
    dist: torch.Tensor,
    known: torch.Tensor,
    resolution: float,
    goal: torch.Tensor,
    lane_bearing: torch.Tensor | None = None,
  ) -> dict:
    """Cost of ``plans [n, k, T, 3]`` under the dynamics model given
    ``inputs``, and the MPPI weights. ``dist``: the obstacle distance field."""
    c, w = self.cfg, self.cfg.weights
    path = self._path(plans)
    pred = self.model.predict(inputs, plans, path)
    pose = path + pred["drift"]
    pen, unknown = sweep(dist, known, pose, pred["drift_std"], resolution, c.footprint)
    to_goal = (pose[..., :2] - goal[:, None, None]).norm(dim=-1)  # [n, k, T]
    terms = {
      "goal": w.goal * to_goal.amin(-1) + w.goal_running * to_goal.mean(-1),
      "obstacle": w.obstacle * pen.sum((-1, -2)),
      "unknown": w.unknown * unknown.float().mean((-1, -2)),
      "fall": w.fall * pred["p_fallen"][..., -1],
      "collision": w.collision * pred["p_collision"].amax(-1),
      "smooth": w.smooth * plans.diff(dim=-2).square().sum((-1, -2)),
    }
    if lane_bearing is not None:
      b = lane_bearing[:, None, None]
      rel = pose[..., :2] - goal[:, None, None]
      lateral = -torch.sin(b) * rel[..., 0] + torch.cos(b) * rel[..., 1]
      terms["lane"] = w.lane * lateral.square().mean(-1)
    cost = torch.stack(list(terms.values())).sum(0)  # [n, k]
    low = cost.min(-1, keepdim=True).values
    gap = cost.median(-1, keepdim=True).values - low
    z = (cost - low) / (c.temperature * gap + 1e-6)
    weight = torch.softmax(-z, dim=-1)
    return {
      "path": path,
      "pose": pose,
      "pred": pred,
      "terms": terms,
      "cost": cost,
      "weight": weight,
      "plan": deadband((weight[..., None, None] * plans).sum(1), c.zero_below),
    }

  @torch.no_grad()
  def step(
    self,
    inputs: dict[str, torch.Tensor],
    height: torch.Tensor,
    known: torch.Tensor,
    resolution: float,
    goal: torch.Tensor,
    lane_bearing: torch.Tensor | None = None,
    explain_inputs: dict[str, torch.Tensor] | None = None,
  ) -> dict[str, Any]:
    """Plan one cycle. ``inputs``: the model inputs; ``height`` / ``known
    [n, H, W]``: the planning map (NaN = unknown); ``goal [n, 2]``,
    ``lane_bearing [n]``: as ``PlanSampler.sample``. Returns ``cmd [n, 3]``
    (to execute now), the new ``plan``, its ``path`` and what the model
    predicts for it (``pred_path`` / ``pred_std``, ``p_fallen`` at the end
    and ``pred_p_fallen`` / ``pred_p_collision`` per step,
    ``pred_contact`` maps per second), and ``cost`` terms.

    With ``explain_inputs`` (the same robots, other model inputs, e.g. another
    fault vector), also ``explain``: every candidate scored under both, and
    the plan MPPI would have chosen under the other inputs."""
    c = self.cfg
    warm = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], dim=1)
    c_now = warm[:, 0] if c.delay_steps else self.plan[:, 0]
    plans = self.sampler.sample(warm, c_now, goal, lane_bearing, c.delay_steps)
    plans = deadband(plans, c.zero_below)
    fp = c.footprint
    dist = distance_field(
      obstacle_cells(height, fp.wall_jump_m), resolution, fp.max_dist_m
    )
    sc = self.score(inputs, plans, dist, known, resolution, goal, lane_bearing)
    path, pose, pred = sc["path"], sc["pose"], sc["pred"]
    terms, cost, weight = sc["terms"], sc["cost"], sc["weight"]
    self.plan = sc["plan"]

    mine = self.plan[:, None]
    mine_path = self._path(mine)
    mine_pred = self.model.predict(inputs, mine, mine_path, contact_maps=True)
    progress = pose[..., -1, :2] @ torch.tensor([1.0, 0.0], device=pose.device)
    probe = {
      f"cruise{vx}_{name}": v[:, 3 + i]
      for i, vx in enumerate(CRUISE_VX)
      for name, v in (
        ("cost", cost),
        ("progress", progress),
        ("p_fallen", pred["p_fallen"][..., -1]),
      )
    }
    best = weight.argmax(-1)
    rows = torch.arange(len(best), device=best.device)
    out = {
      "cmd": self.plan[:, 0],
      "plan": self.plan,
      "path": mine_path[:, 0],
      "pred_path": (mine_path + mine_pred["drift"])[:, 0],
      "pred_std": mine_pred["drift_std"][:, 0],
      "p_fallen": mine_pred["p_fallen"][:, 0, -1],
      "pred_p_fallen": mine_pred["p_fallen"][:, 0],
      "pred_p_collision": mine_pred["p_collision"][:, 0],
      "pred_contact": mine_pred["contact"][:, 0],
      "ess": 1.0 / weight.square().sum(-1),
      "cost_best": cost.min(-1).values,
      **{f"cost_{k}": v[rows, best] for k, v in terms.items()},
      **{f"probe_{k}": v for k, v in probe.items()},
    }
    if explain_inputs is not None:
      alt = self.score(
        explain_inputs, plans, dist, known, resolution, goal, lane_bearing
      )
      alt_path = self._path(alt["plan"][:, None])
      alt_pred = self.model.predict(explain_inputs, alt["plan"][:, None], alt_path)
      # The chosen plan under the other inputs, and the alternative's choice
      # under these, so each choice can be read under both models.
      cross = self.model.predict(inputs, alt["plan"][:, None], alt_path)
      cross_mine = self.model.predict(explain_inputs, mine, mine_path)
      out["explain"] = {
        "plans": plans,
        "path": path,
        "pose": pose,
        "cost": cost,
        "weight": weight,
        "terms": terms,
        "alt_pose": alt["pose"],
        "alt_cost": alt["cost"],
        "alt_weight": alt["weight"],
        "alt_terms": alt["terms"],
        "alt_plan": alt["plan"],
        "alt_plan_path": alt_path[:, 0],
        "alt_plan_pred": (alt_path + alt_pred["drift"])[:, 0],
        "alt_plan_pred_here": (alt_path + cross["drift"])[:, 0],
        "plan_pred_alt": (mine_path + cross_mine["drift"])[:, 0],
      }
    return out
