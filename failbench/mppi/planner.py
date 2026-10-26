"""MPPI over the impact model.

One cycle (every ``step_s``) for ``n`` robots:

1. sample ``k`` plans per robot (``PlanSampler``), deadbanded as the
   command term will apply them, and integrate their commanded paths;
2. predict drift (and its spread), falls and trunk contact for every plan
   with the dynamics model (one encode per robot);
3. cost each plan: progress to the goal, the footprint sweep of the
   predicted path ``cmd_path + drift +- k sigma`` over the obstacle map,
   and over drops (ledges), unknown cells, P(fallen by the end), P(trunk
   contact), smoothness, the fault risk (below), going
   nowhere once the robot has been still a while, and on the course the
   offset from the lane centre;
   The fault risk (``FaultRiskCfg``, off at ``p = 0``) prices a fault that
   hasn't happened: what each single-joint fault would do if it fired at
   points along the plan's predicted path, the robot carrying on at the
   plan's speed (it hasn't noticed yet, and couldn't stop at once if it
   had). Each branch's throw (drift, spread and falls) is predicted once
   per robot from the current state, as if it fired now, at a few held
   speeds (``hold_vx``), then placed at each firing point's pose, from the
   speed nearest the plan's there. The plan pays, summed over branches weighted by
   their priors and averaged over firing points, each obstacle's severity
   times the chance the thrown robot touches it (``hit_prob``) or falls
   onto it, whichever is larger. Severity comes per map cell
   (``failbench/severity.py``; drops have their own), so a healthy robot
   keeps away from where a fault would throw it, from glass more than a
   box; in open ground nothing is in reach and the risk is 0.
   Every hazard (obstacle, drop, fall, collision, fault risk) is discounted
   by ``discount_per_s`` per second ahead, so a nearer hit costs more.
4. weight plans by ``exp(-(cost - min) / (temperature * (q - min)))``, ``q``
   the ``gap_quantile`` cost (the median by default), and average them into
   the new plan, kept (shifted) as the next warm start. Scaling by a
   quantile's gap, not the spread, keeps a few hopeless plans (stops, holds
   into a wall) from flattening the weights; a low quantile lets small cost
   differences count when many plans are hopeless.

With ``delay_steps`` the first steps of every plan are the commands already
committed: on the robot, a plan computed from the state at ``t`` starts
executing one cycle later, and the sim runs the same way.

Inputs are in each robot's current yaw frame (origin at the trunk, +x
forward): the goal, the lane, and the map, which shares the frame of the
plans' predicted paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

import torch

from failbench.faults import JOINT_NAMES, LOCK, NOMINAL, TORQUE, encode_fault
from failbench.mppi.costmap import (
  FootprintCfg,
  distance_field,
  drop_cells,
  hit_prob,
  obstacle_cells,
  sweep,
)
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
  drop: float = 100.0
  """Per metre of footprint penetration into a drop (``drop_cells``), as
  ``obstacle``: going over a ledge is worse than bumping a wall."""
  unknown: float = 0.2
  """Per footprint circle on an unknown cell, averaged over steps."""
  fall: float = 20.0
  collision: float = 5.0
  smooth: float = 0.5
  """Per (m/s)^2 or (rad/s)^2 of change between plan steps, summed."""
  idle: float = 4.0
  """For a plan that moves the robot nowhere, once the robot has been still
  long enough (``MppiCfg.idle_after_s``, ``idle_ramp_s``); less the further
  the plan moves it, nothing from ``idle_move_m``. Standing otherwise costs
  only the goal term, while every plan that moves also pays its risks, so
  near a waypoint or with a bad leg standing can win every cycle: nothing
  changes while the robot stands, so it stands for good."""
  fault_risk: float = 1.0
  """On ``FaultRiskCfg``'s expected severity: how much a fault that hasn't
  happened matters, apart from how likely it is (``p``)."""
  discount_per_s: float = 1.0
  """Hazards (obstacle, drop, fall, collision, fault risk) are weighted by
  this to the power of how many seconds ahead they happen, as RL's gamma: a
  hit in 1 s costs more than the same hit in 5 s. 1 weighs every step alike."""


VULNERABLE_FAULTS = (
  "FL_hip_joint:free",
  "FR_hip_joint:free",
  "RL_thigh_joint:free",
  "RR_thigh_joint:free",
  "FL_calf_joint:lock",
  "FR_calf_joint:lock",
  "RL_thigh_joint:lock",
  "RR_thigh_joint:lock",
)
"""Each leg's worst limp and worst seized joint in the data (from the fault
time, by falls within 5 s, then by throw): limp front hips fell 56-58% of
the time, front calves seizing 14-32%. Rear seizures barely fall (1-2%) so
the rear pick is the one that throws furthest, the thigh (1.2-1.4 m by
5 s); RR's limp hip and thigh are close (28% vs 30%), thigh kept for
symmetry. Weak (partial torque) faults fell at most 4% and aren't priced."""


@dataclass(frozen=True)
class FaultRiskCfg:
  p: float = 0.0
  """Probability a fault fires during the plan, split over the branches
  (``mode_share``, then evenly over each mode's joints) and spread evenly
  over the firing points; 0 turns the term off."""
  faults: tuple[str, ...] | None = VULNERABLE_FAULTS
  """Branches as ``joint:mode`` pairs, a joint going limp (``free``,
  torque 0) or seizing (``lock``); None: every joint in each of
  ``modes``."""
  modes: tuple[str, ...] = ("free", "lock")
  """The modes priced; ``faults`` in other modes are skipped."""
  measure: Literal["mean", "cvar", "max"] = "mean"
  """How the outcomes (a fault firing at a point, each with probability
  prior / F; no fault, probability 1 - p, costs 0) become one cost:
  ``mean`` the expected severity; ``cvar`` the mean of the worst
  ``cvar_alpha`` of the probability (1: the mean over everything, ``p``:
  the mean given a fault happens, toward 0: the worst); ``max`` the worst
  outcome, however unlikely."""
  cvar_alpha: float = 0.025
  """CVaR's tail mass; with ``p`` 0.1 the default is the worst quarter of
  the fault outcomes."""
  mode_share: tuple[float, ...] | None = None
  """Each mode's share of ``p`` (normalised); None: equal."""
  fire_s: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
  """Firing points, s into the plan."""
  post_s: float = 2.5
  """How long after firing the throw is swept. In the data a free hip
  throws the robot 0.2-0.4 m toward its side by 1.5 s and 0.4-0.6 m by
  2 s, and most falls come within 2 s of the fault."""
  hold_vx: tuple[float, ...] = (0.3, 0.6, 0.9)
  """Forward speeds the robot is assumed to hold after a fault fires; each
  firing point uses the nearest to the plan's speed there. Holding the
  sample's own command reproduces the data's free-hip throws (within a few
  cm at 1.5 s); assuming a stop halves them."""
  fallen_radius: float = 0.4
  """Footprint circles' radius for a robot lying down (``hit_prob``)."""
  default_severity: float = 10.0
  """Obstacle severity when the planner is given no severity map."""
  drop_severity: float = 50.0
  """Severity of going over a drop (``drop_cells``)."""


def risk_weights(
  cost: torch.Tensor, prob: torch.Tensor, fr: FaultRiskCfg
) -> torch.Tensor:
  """Weights over outcomes ``cost [..., O]`` (probabilities ``prob [O]``,
  summing to ``p``; no fault, the rest, costs 0) whose dot with the costs is
  ``FaultRiskCfg.measure``: the probabilities (the expectation), the CVaR's
  tail mass over ``cvar_alpha``, or one-hot on the worst outcome."""
  if fr.measure == "mean":
    return prob.expand_as(cost)
  if fr.measure == "max":
    return torch.nn.functional.one_hot(cost.argmax(-1), cost.shape[-1]).to(cost)
  if fr.measure != "cvar":
    raise ValueError(f"Unknown risk measure {fr.measure!r}.")
  # Fill the worst cvar_alpha of the probability, worst outcome first.
  order = cost.argsort(-1, descending=True)
  q = prob[order]
  before = q.cumsum(-1) - q
  taken = (fr.cvar_alpha - before).clamp(min=0).minimum(q)
  return torch.zeros_like(cost).scatter(-1, order, taken / fr.cvar_alpha)


@dataclass(frozen=True)
class MppiCfg:
  sampler: SamplerCfg = field(default_factory=SamplerCfg)
  footprint: FootprintCfg = field(default_factory=FootprintCfg)
  weights: CostWeights = field(default_factory=CostWeights)
  fault_risk: FaultRiskCfg = field(default_factory=FaultRiskCfg)
  temperature: float = 0.25
  gap_quantile: float = 0.5
  """Weights scale by the gap from the best plan to this cost quantile."""
  control_dt: float = 0.02
  """Control step the plans are held over (HIM runs at 50 Hz)."""
  delay_steps: int = 1
  zero_below: float = 0.2
  """The command term's deadband (``GoalVelocityCommandCfg.zero_below``)."""
  route_switch_m: float = 1.0
  """A predicted path within this of the goal has passed it (``route``)."""
  still_radius_m: float = 0.15
  """The robot is still while it stays within this of where it stopped."""
  idle_after_s: float = 2.0
  idle_ramp_s: float = 4.0
  """The idle cost starts once the robot has been still ``idle_after_s`` and
  reaches its full weight ``idle_ramp_s`` later."""
  idle_move_m: float = 0.5
  """A plan whose predicted path gets this far from the robot isn't idle."""
  fall_soon_s: float = 1.0
  """``p_fall_soon``: P(fall within this) on the chosen plan, for a monitor."""


class Mppi:
  def __init__(self, cfg: MppiCfg, model: DynamicsModel, n: int, device):
    self.cfg = cfg
    self.model = model
    self.sampler = PlanSampler(cfg.sampler, device)
    s = cfg.sampler
    self.substeps = round(s.step_s / cfg.control_dt)
    self.plan = torch.zeros(n, s.steps, 3, device=device)
    """Each robot's current plan; ``plan[:, 0]`` runs now."""
    self.still_s = torch.zeros(n, device=device)
    """How long each robot has stayed within ``still_radius_m`` of
    ``_anchor`` (``track``)."""
    self._anchor = torch.full((n, 2), float("nan"), device=device)

  def reset(self, ids: torch.Tensor, cmd: torch.Tensor | None = None) -> None:
    self.plan[ids] = 0.0 if cmd is None else cmd[:, None]
    self.still_s[ids] = 0.0
    self._anchor[ids] = float("nan")

  def track(self, pos_xy: torch.Tensor, dt: float) -> None:
    """Update ``still_s`` from the robots' positions ``[n, 2]`` (any fixed
    frame), ``dt`` after the last call."""
    moved = ~((pos_xy - self._anchor).norm(dim=-1) < self.cfg.still_radius_m)
    self._anchor = torch.where(moved[:, None], pos_xy, self._anchor)
    self.still_s = torch.where(moved, 0.0, self.still_s + dt)

  def advance(self) -> torch.Tensor:
    """Move one step along the current plans without replanning (a planner
    that commits to a plan for several cycles); returns the commands to run
    now."""
    self.plan = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], dim=1)
    return self.plan[:, 0]

  def _pairs(self, num_joints: int) -> list[tuple[int, str]]:
    """The fault branches as (joint index, mode)."""
    fr = self.cfg.fault_risk
    if fr.faults is None:
      pairs = [(j, m) for m in fr.modes for j in range(num_joints)]
    else:
      if num_joints != len(JOINT_NAMES):
        raise ValueError(f"Named faults need the {len(JOINT_NAMES)} Go2 joints.")
      split = [f.split(":") for f in fr.faults]
      pairs = [(JOINT_NAMES.index(j), m) for j, m in split if m in fr.modes]
    for _, mode in pairs:
      if mode not in ("free", "lock"):
        raise ValueError(f"Unknown fault mode {mode!r}.")
    return pairs

  def branch_names(self, num_joints: int = len(JOINT_NAMES)) -> list[str]:
    """``joint:mode`` per fault branch, in ``risk_branch`` order."""
    names = JOINT_NAMES if num_joints == len(JOINT_NAMES) else range(num_joints)
    return [f"{names[j]}:{m}" for j, m in self._pairs(num_joints)]

  def _branches(self, num_joints: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Encoded single-joint faults ``[B, J * 2]`` (``FaultRiskCfg.faults``
    in ``modes``) and their priors ``[B]`` (summing to ``p``)."""
    fr = self.cfg.fault_risk
    share = fr.mode_share or [1.0] * len(fr.modes)
    if len(share) != len(fr.modes):
      raise ValueError("mode_share needs one share per mode.")
    pairs = self._pairs(num_joints)
    per_mode = {m: sum(pm == m for _, pm in pairs) for m in fr.modes}
    total = sum(share[i] for i, m in enumerate(fr.modes) if per_mode[m])
    prior = [fr.p * share[fr.modes.index(m)] / total / per_mode[m] for _, m in pairs]
    params = torch.tensor(NOMINAL, device=device).repeat(len(pairs), num_joints, 1)
    for b, (j, mode) in enumerate(pairs):
      params[b, j, TORQUE if mode == "free" else LOCK] = 0.0 if mode == "free" else 1.0
    return encode_fault(params), torch.tensor(prior, device=device)

  def _throws(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Each fault branch fired now, the robot holding each ``hold_vx``:
    ``thrown`` path ``[n, V, B, Tp, 3]`` (robot frame), its ``std`` and
    ``p_fall``, and the branches' ``prior [B]``. One model call per cycle,
    shared by every plan."""
    c, fr = self.cfg, self.cfg.fault_risk
    s = c.sampler
    n = inputs["fault"].shape[0]
    device = inputs["fault"].device
    branches, prior = self._branches(inputs["fault"].shape[-1] // 2, device)
    nb = len(branches)
    vxs = torch.tensor(fr.hold_vx, device=device)
    nv = len(vxs)
    # Each branch, fired now, the robot holding each speed: [n, V, B] rows.
    rep = {key: v.repeat_interleave(nv * nb, dim=0) for key, v in inputs.items()}
    rep["fault"] = branches.repeat(n * nv, 1).to(inputs["fault"].dtype)
    held = torch.zeros(n, nv, nb, 1, s.steps, 3, device=device)
    held[..., 0] = vxs[None, :, None, None, None]
    held = held.flatten(0, 2)
    throw = self.model.predict(rep, held, self._path(held))
    tp = round(fr.post_s / s.step_s)
    # Drift is off the held command's own path; the throw is both.
    thrown = (self._path(held) + throw["drift"])[:, 0, :tp]
    return {
      "thrown": thrown.unflatten(0, (n, nv, nb)),
      "std": throw["drift_std"][:, 0, :tp].unflatten(0, (n, nv, nb)),
      "p_fall": throw["p_fallen"][:, 0, :tp].unflatten(0, (n, nv, nb)),
      "prior": prior,
    }

  def _fault_risk(
    self,
    throws: dict[str, torch.Tensor],
    plans: torch.Tensor,
    pose: torch.Tensor,
    fields: list[tuple[float, torch.Tensor]],
    resolution: float,
    per_branch: bool = False,
  ) -> torch.Tensor | dict[str, torch.Tensor]:
    """Expected severity of a fault's throw per plan ``[n, k]``: over
    branches (by prior) and firing points (mean), over ``fields`` (severity,
    distance field), severity times P(touch or fall onto). ``per_branch``:
    instead each branch's share ``cost [n, k, B]`` (prior-weighted; they sum
    to the total) and the throws ``placed`` at the firing points
    ``[n, k, F, B, Tp, 3]`` (map frame) with their ``std`` and ``p_fall``."""
    c, fr = self.cfg, self.cfg.fault_risk
    n, k = pose.shape[:2]
    s = c.sampler
    prior = throws["prior"]
    nb = len(prior)
    vxs = torch.tensor(fr.hold_vx, device=pose.device)
    d, sd, fall = throws["thrown"], throws["std"], throws["p_fall"]
    tp = d.shape[-2]
    # Placed at each firing point's pose on every plan, from the held speed
    # nearest the plan's there.
    at = torch.tensor([round(t / s.step_s) - 1 for t in fr.fire_s], device=pose.device)
    p0 = pose[:, :, at]  # [n, k, F, 3]
    near = (plans[:, :, at, None, 0] - vxs).abs().argmin(-1)  # [n, k, F]
    rows = torch.arange(n, device=pose.device)[:, None, None]
    d, sd, fall = d[rows, near], sd[rows, near], fall[rows, near]  # [n, k, F, B, ...]
    yaw = p0[..., None, None, 2]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    dx, dy = d[..., 0], d[..., 1]
    placed = torch.stack(
      [
        p0[..., None, None, 0] + cy * dx - sy * dy,
        p0[..., None, None, 1] + sy * dx + cy * dy,
        yaw + d[..., 2],
      ],
      dim=-1,
    )  # [n, k, F, B, Tp, 3]
    nf = len(at)
    flat = placed.flatten(1, 3)
    std = sd.flatten(1, 3)
    p_fall = fall.flatten(1, 3)
    # Discounted by when the hit would happen: firing time plus time since.
    after = (torch.arange(tp, device=pose.device) + 1) * s.step_s
    when = torch.tensor(fr.fire_s, device=pose.device)[:, None] + after
    disc = (c.weights.discount_per_s**when)[:, None]  # [F, 1, Tp]
    cost = torch.zeros(n, k, nf, nb, device=pose.device)
    for severity, dist in fields:
      touch = hit_prob(dist, flat, std, resolution, c.footprint)
      onto = p_fall * hit_prob(
        dist, flat, std, resolution, c.footprint, radius=fr.fallen_radius
      )
      hit = torch.maximum(touch, onto).view(n, k, nf, nb, tp)
      cost = cost + severity * (disc * hit).amax(-1)
    # Outcomes (firing point, fault), each as likely as prior / F; the rest
    # of the probability (no fault) costs nothing.
    outcome = cost.flatten(2)  # [n, k, F * B]
    part = risk_weights(outcome, prior.repeat(nf) / nf, fr) * outcome
    if per_branch:
      return {
        "cost": part.view(n, k, nf, nb).sum(2),
        "placed": placed,
        "std": sd,
        "p_fall": fall,
      }
    return part.sum(-1)

  def _risk_fields(
    self,
    obst: torch.Tensor,
    drops: torch.Tensor,
    severity: torch.Tensor | None,
    resolution: float,
  ) -> list[tuple[float, torch.Tensor]]:
    """(severity, distance field) for each severity level on the maps:
    obstacle cells at their ``severity``, drops at ``drop_severity``."""
    fr, fp = self.cfg.fault_risk, self.cfg.footprint
    level = torch.where(
      obst,
      fr.default_severity if severity is None else severity,
      0.0,
    )
    level = torch.where(drops, torch.clamp(level, min=fr.drop_severity), level)
    return [
      (v, distance_field(level == v, resolution, fp.max_dist_m))
      for v in level.unique().tolist()
      if v > 0
    ]

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
    route: torch.Tensor | None = None,
    drop_dist: torch.Tensor | None = None,
    exposed: torch.Tensor | None = None,
    risk_fields: list[tuple[float, torch.Tensor]] | None = None,
  ) -> dict:
    """Cost of ``plans [n, k, T, 3]`` under the dynamics model given
    ``inputs``, and the MPPI weights. ``dist`` / ``drop_dist``: the obstacle
    and drop distance fields (no drops without the latter). ``exposed [n]``:
    robots the fault risk applies to (still healthy as far as the planner
    knows; all by default). ``risk_fields``: (severity, distance field) per
    severity level on the map, for the fault risk.
    With ``route [n, 2]`` (the waypoint after ``goal``), the goal terms use
    the remaining route length: to ``goal`` plus on to ``route``, or once
    the predicted path has passed within ``route_switch_m`` of ``goal``,
    straight to ``route``."""
    c, w = self.cfg, self.cfg.weights
    path = self._path(plans)
    pred = self.model.predict(inputs, plans, path)
    pose = path + pred["drift"]
    pen, unknown = sweep(dist, known, pose, pred["drift_std"], resolution, c.footprint)
    if drop_dist is None:
      drop_pen = torch.zeros_like(pen)
    else:
      drop_pen, _ = sweep(
        drop_dist, known, pose, pred["drift_std"], resolution, c.footprint
      )
    to_goal = (pose[..., :2] - goal[:, None, None]).norm(dim=-1)  # [n, k, T]
    if route is not None:
      passed = torch.cummax((to_goal < c.route_switch_m).float(), dim=-1).values > 0
      on_from = (route - goal).norm(dim=-1)[:, None, None]
      to_route = (pose[..., :2] - route[:, None, None]).norm(dim=-1)
      to_goal = torch.where(passed, to_route, to_goal + on_from)
    steps = plans.shape[-2]
    ahead = (torch.arange(steps, device=plans.device) + 1) * c.sampler.step_s
    disc = w.discount_per_s**ahead  # [T]
    # P(falling during each step), so a fall is charged when it happens.
    fallen = pred["p_fallen"].cummax(dim=-1).values
    falls = fallen.diff(dim=-1, prepend=torch.zeros_like(fallen[..., :1]))
    terms = {
      "goal": w.goal * to_goal.amin(-1) + w.goal_running * to_goal.mean(-1),
      "obstacle": w.obstacle * (disc * pen.sum(-1)).sum(-1),
      "drop": w.drop * (disc * drop_pen.sum(-1)).sum(-1),
      "unknown": w.unknown * unknown.float().mean((-1, -2)),
      "fall": w.fall * (disc * falls).sum(-1),
      "collision": w.collision
      * (disc * pred.get("p_collision", torch.zeros_like(fallen))).amax(-1),
      "smooth": w.smooth * plans.diff(dim=-2).square().sum((-1, -2)),
      "idle": w.idle
      * ((self.still_s - c.idle_after_s) / c.idle_ramp_s).clamp(0, 1)[:, None]
      * (1 - pose[..., :2].norm(dim=-1).amax(-1) / c.idle_move_m).clamp(min=0),
    }
    throws = None
    if c.fault_risk.p > 0:
      if risk_fields:
        throws = self._throws(inputs)
        risk = self._fault_risk(throws, plans, pose, risk_fields, resolution)
        risk = w.fault_risk * cast(torch.Tensor, risk)
      else:
        risk = pose.new_zeros(pose.shape[:2])
      if exposed is not None:
        risk = risk * exposed[:, None]
      terms["fault_risk"] = risk
    if lane_bearing is not None:
      b = lane_bearing[:, None, None]
      rel = pose[..., :2] - goal[:, None, None]
      lateral = -torch.sin(b) * rel[..., 0] + torch.cos(b) * rel[..., 1]
      terms["lane"] = w.lane * lateral.square().mean(-1)
    cost = torch.stack(list(terms.values())).sum(0)  # [n, k]
    low = cost.min(-1, keepdim=True).values
    gap = cost.quantile(c.gap_quantile, dim=-1, keepdim=True) - low
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
      "throws": throws,
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
    route: torch.Tensor | None = None,
    exposed: torch.Tensor | None = None,
    severity: torch.Tensor | None = None,
  ) -> dict[str, Any]:
    """Plan one cycle. ``inputs``: the model inputs; ``height`` / ``known
    [n, H, W]``: the planning map (NaN = unknown); ``severity [n, H, W]``:
    what hitting each cell would cost, if an obstacle (``FaultRiskCfg``'s
    default for all without it); ``goal [n, 2]``,
    ``lane_bearing [n]``: as ``PlanSampler.sample``. Returns ``cmd [n, 3]``
    (to execute now), the new ``plan``, its ``path`` and what the model
    predicts for it (``pred_path`` / ``pred_std``, ``p_fallen`` at the end
    and ``pred_p_fallen`` / ``pred_p_collision`` per step, ``p_fall_soon``
    within ``fall_soon_s``, and with a contact-head model ``pred_contact``
    maps per second), and ``cost`` terms.

    With ``explain_inputs`` (the same robots, other model inputs, e.g. another
    fault vector), also ``explain``: every candidate scored under both, and
    the plan MPPI would have chosen under the other inputs."""
    c = self.cfg
    warm = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], dim=1)
    c_now = warm[:, 0] if c.delay_steps else self.plan[:, 0]
    plans = self.sampler.sample(warm, c_now, goal, lane_bearing, c.delay_steps, route)
    plans = deadband(plans, c.zero_below)
    fp = c.footprint
    obst = obstacle_cells(height, fp.wall_jump_m)
    drops = drop_cells(height, fp, resolution)
    dist = distance_field(obst, resolution, fp.max_dist_m)
    drop_dist = distance_field(drops, resolution, fp.max_dist_m)
    risk_fields = None
    if c.fault_risk.p > 0:
      risk_fields = self._risk_fields(obst, drops, severity, resolution)
    sc = self.score(
      inputs,
      plans,
      dist,
      known,
      resolution,
      goal,
      lane_bearing,
      route,
      drop_dist,
      exposed,
      risk_fields,
    )
    path, pose, pred = sc["path"], sc["pose"], sc["pred"]
    terms, cost, weight = sc["terms"], sc["cost"], sc["weight"]
    self.plan = sc["plan"]
    # Plans that each move, but in different directions, can average below
    # the deadband: the robot gets a zero command the idle cost never sees.
    # Once it has been still that long, run the best single plan instead.
    rows = torch.arange(len(cost), device=cost.device)
    best_plan = plans[rows, cost.argmin(-1)]
    gets_nowhere = (
      self._path(self.plan[:, None])[:, 0, :, :2].norm(dim=-1).amax(-1) < c.idle_move_m
    )
    swap = gets_nowhere & (self.still_s >= c.idle_after_s)
    self.plan = torch.where(swap[:, None, None], best_plan, self.plan)

    mine = self.plan[:, None]
    mine_path = self._path(mine)
    mine_pred = self.model.predict(inputs, mine, mine_path, contact_maps=True)
    s_step = c.sampler.step_s
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
    out = {
      "cmd": self.plan[:, 0],
      "plan": self.plan,
      "path": mine_path[:, 0],
      "pred_path": (mine_path + mine_pred["drift"])[:, 0],
      "pred_std": mine_pred["drift_std"][:, 0],
      "p_fallen": mine_pred["p_fallen"][:, 0, -1],
      "pred_p_fallen": mine_pred["p_fallen"][:, 0],
      "pred_p_collision": mine_pred.get(
        "p_collision", torch.zeros_like(mine_pred["p_fallen"])
      )[:, 0],
      "p_fall_soon": mine_pred["p_fallen"][:, 0, round(c.fall_soon_s / s_step) - 1],
      "ess": 1.0 / weight.square().sum(-1),
      "cost_best": cost.min(-1).values,
      **{f"cost_{k}": v[rows, best] for k, v in terms.items()},
      **{f"probe_{k}": v for k, v in probe.items()},
    }
    if "contact" in mine_pred:
      out["pred_contact"] = mine_pred["contact"][:, 0]
    if sc["throws"] is not None and risk_fields:
      # What each fault branch would do to the plan being run, for logs.
      mine_pose = mine_path + mine_pred["drift"]
      br = self._fault_risk(
        sc["throws"], mine, mine_pose, risk_fields, resolution, per_branch=True
      )
      br = cast(dict[str, torch.Tensor], br)
      scale = c.weights.fault_risk
      if exposed is not None:
        scale = scale * exposed[:, None]
      out["risk_branch"] = scale * br["cost"][:, 0]
      out["risk_thrown"] = br["placed"][:, 0]
      out["risk_thrown_std"] = br["std"][:, 0]
      out["risk_p_fall"] = br["p_fall"][:, 0]
    if explain_inputs is not None:
      alt = self.score(
        explain_inputs,
        plans,
        dist,
        known,
        resolution,
        goal,
        lane_bearing,
        route,
        drop_dist,
        exposed,
        risk_fields,
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
