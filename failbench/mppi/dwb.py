"""Nav2's DWB local planner (``dwb_core::DWBLocalPlanner``), ported to batched
torch: the second stock planner every FailBench arm can share.

One cycle, per robot:

1. the dynamic window (``XYThetaIterator``): every axis's reachable speeds
   in one control period (``sim_period``) from the current speed, inside the limits,
   ``vx_samples x vy_samples x wz_samples`` twists;
2. each twist held for ``sim_time`` and walked toward under the
   acceleration limits (``StandardTrajectoryGenerator``), integrated;
3. scored by the critics of ``nav2_bringup/params/nav2_params.yaml`` with
   their default scales; a twist any critic rules out is dropped;
4. the cheapest twist is sent; none left, the robot stops.

What differs from Nav2 (``DwbCfg``): the Go2's forward and turning speed
limits (Nav2's diff-drive kinematics kept: no sidestep), FailBench's 10 Hz
planning cycle and the motion model shared with ``failbench/mppi/nav2.py``
(the HIM policy's sidestep gain and deadband) at fixed ``model_dt`` steps
instead of Nav2's distance granularity. There is no global planner: the path
is the straight line to the goal (as in ``nav2.py``), so the path and goal
distance critics measure Euclidean distance to it, where Nav2 grows a
wavefront over the costmap; on a straight path in free space they agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from failbench.mppi.nav2 import (
  INSCRIBED,
  LETHAL,
  NO_INFORMATION,
  CostmapCfg,
  _lookup,
  _wrap,
  accel_limited,
  footprint_max_cost,
  integrate,
  straight_path,
)


@dataclass(frozen=True)
class DwbCfg:
  vx_min: float = 0.0
  vx_max: float = 1.2
  """The HIM policy's range; no walking backwards (Nav2: 0 to 0.26)."""
  vy_max: float = 0.0
  """Nav2's stock DWB is a diff-drive's (no sidestep). Its critics score
  where a rollout ends, not how the body faces its travel, so with
  sidesteps it crabs (body turned one way, strafing the other) along the
  re-anchored straight path until a footprint corner clips an obstacle."""
  wz_max: float = 1.0
  ax_max: float = 2.5
  ax_min: float = -2.5
  ay_max: float = 2.5
  """Braking a measured sideways drift to 0 (Nav2: 0)."""
  ay_min: float = -2.5
  az_max: float = 3.2
  vx_samples: int = 20
  vy_samples: int = 1
  """Nav2: 5, moot at ``vy_max`` 0."""
  wz_samples: int = 20
  sim_time: float = 1.7
  sim_period: float = 0.1
  """1 / ``controller_frequency``, the dynamic window's span: FailBench's
  10 Hz cycle (Nav2: 20 Hz). At Nav2's 0.05 s a robot at rest can't reach
  the Go2's 0.2 m/s deadband in one window and never sets off."""
  model_dt: float = 0.1
  """FailBench's 10 Hz planning cycle, the rollout step."""
  min_speed_xy: float = 0.0
  min_speed_theta: float = 0.0
  xy_goal_tolerance: float = 0.25
  trans_stopped_velocity: float = 0.25
  base_obstacle_scale: float = 0.02
  path_align_scale: float = 32.0
  goal_align_scale: float = 24.0
  path_dist_scale: float = 32.0
  goal_dist_scale: float = 24.0
  rotate_to_goal_scale: float = 32.0
  forward_point_distance: float = 0.1
  rotate_slowing_factor: float = 5.0
  rotate_lookahead_time: float = -1.0
  oscillation_reset_dist: float = 0.05
  oscillation_reset_angle: float = 0.2
  vy_gain: float = 0.7
  """The shared motion model (``Nav2MppiCfg``): the Go2 walks about 0.71 of
  a commanded sidestep."""
  deadband: float = 0.2
  prune_distance: float = 4.0
  path_resolution: float = 0.05
  plan_steps: int = 50
  commit_steps: int = 30
  """A twist held this many steps (3 s): the candidate the fault-risk
  critic and the data price."""
  costmap: CostmapCfg = field(default_factory=CostmapCfg)


def velocity_grid(v0: torch.Tensor, cfg: DwbCfg) -> torch.Tensor:
  """The dynamic window's twists ``[n, S, 3]`` from speeds ``v0 [n, 3]``:
  per axis ``vx_samples`` (``vy``, ``wz``) evenly from the slowest to the
  fastest speed one ``sim_period`` of acceleration reaches, inside the
  limits; every combination."""
  dt = cfg.sim_period
  axes = []
  for i, (lo, hi, dec, acc, count) in enumerate(
    (
      (cfg.vx_min, cfg.vx_max, cfg.ax_min, cfg.ax_max, cfg.vx_samples),
      (-cfg.vy_max, cfg.vy_max, cfg.ay_min, cfg.ay_max, cfg.vy_samples),
      (-cfg.wz_max, cfg.wz_max, -cfg.az_max, cfg.az_max, cfg.wz_samples),
    )
  ):
    v = v0[:, i]
    low = (v + dec * dt).clamp(lo, hi)
    high = (v + acc * dt).clamp(lo, hi)
    s = torch.linspace(0, 1, count, device=v0.device)
    axes.append(low[:, None] + (high - low)[:, None] * s)  # [n, count]
  vx, vy, wz = axes
  n = v0.shape[0]
  grid = torch.stack(
    [
      vx[:, :, None, None].expand(-1, -1, cfg.vy_samples, cfg.wz_samples),
      vy[:, None, :, None].expand(-1, cfg.vx_samples, -1, cfg.wz_samples),
      wz[:, None, None, :].expand(-1, cfg.vx_samples, cfg.vy_samples, -1),
    ],
    -1,
  )
  return grid.reshape(n, -1, 3)


def rollout(
  twist: torch.Tensor, v0: torch.Tensor, cfg: DwbCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Each twist ``[n, S, 3]`` held for ``sim_time``, walked toward from
  ``v0 [n, 3]`` within the acceleration limits: poses and velocities
  ``[n, S, T, 3]`` from the origin, ``T = sim_time / model_dt``."""
  steps = round(cfg.sim_time / cfg.model_dt)
  cv = twist[:, :, None].expand(-1, -1, steps, -1)
  v = accel_limited(cv, v0, cfg)
  return integrate(v, torch.zeros_like(v0), cfg), v


def _nearest(points: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
  """Distance from points ``[n, S, 2]`` to the nearest of ``path [n, P, 2]``."""
  return (points[:, :, None] - path[:, None]).norm(dim=-1).amin(-1)


class Dwb:
  """Batched Nav2 DWB for ``n`` robots, with ``Nav2Mppi``'s interface (the
  runner's ``reset`` / ``track`` / ``advance``, and ``step``'s outputs).
  ``extra_critics``: callables ``(data) -> cost [n, S]`` added to the
  critics' sum; ``data["cv"]`` holds each twist held for ``commit_steps``."""

  def __init__(self, cfg: DwbCfg, n: int, device, extra_critics=()):
    self.cfg = cfg
    self.n = n
    self.device = device
    self.extra_critics = list(extra_critics)
    self.substeps = 5
    """Control steps per cycle (50 Hz policy, 10 Hz planning)."""
    self.plan = torch.zeros(n, cfg.plan_steps, 3, device=device)
    self.last_cmd = torch.zeros(n, 3, device=device)
    # OscillationCritic's state: each axis's direction of travel, whether
    # it has flipped since the robot last moved enough, and how far it has
    # moved (m, rad) since the flip.
    self.trend = torch.zeros(n, 3, device=device)
    self.flipped = torch.zeros(n, 3, dtype=torch.bool, device=device)
    self.moved = torch.zeros(n, 2, device=device)

  # The runner's planner interface.
  def reset(self, ids: torch.Tensor, cmd: torch.Tensor | None = None) -> None:
    self.plan[ids] = 0.0 if cmd is None else cmd[:, None]
    self.last_cmd[ids] = 0.0 if cmd is None else cmd
    self.trend[ids] = 0.0
    self.flipped[ids] = False
    self.moved[ids] = 0.0

  def track(self, pos_xy: torch.Tensor, dt: float) -> None:
    del pos_xy, dt

  def advance(self) -> torch.Tensor:
    self.plan = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], dim=1)
    return self.plan[:, 0]

  def branch_names(self, num_joints: int = 12) -> list[str]:
    del num_joints
    return []

  @torch.no_grad()
  def step(
    self,
    costmap: torch.Tensor,
    resolution: float,
    goal: torch.Tensor,
    speed: torch.Tensor,
    context: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Plan one cycle, as ``Nav2Mppi.step``: ``costmap [n, H, W]``, ``goal
    [n, 2]`` and ``speed [n, 3]`` (measured vx, vy, wz) in the robot's yaw
    frame. ``weight`` is 1 on the chosen twist."""
    c = self.cfg
    n, dev = self.n, self.device
    self._oscillation_moved(speed)
    twist = velocity_grid(speed, c)  # [n, S, 3]
    traj, v = rollout(twist, speed, c)
    pose0 = torch.zeros(n, 3, device=dev)
    path, path_yaw, path_real, path_len = straight_path(pose0[:, :2], goal, c)
    end = path.gather(1, (path_real - 1)[:, None, None].expand(-1, 1, 2))
    cv = twist[:, :, None].expand(-1, -1, c.commit_steps, -1)
    data = {
      "traj": traj,
      "v": v,
      "cv": cv,
      "pose0": pose0,
      "path": path,
      "path_yaw": path_yaw,
      "path_real": path_real,
      "path_length": path_len,
      "goal": goal,
      "costmap": costmap,
      "resolution": resolution,
      **(context or {}),
    }
    terms, legal = self._critics(data, twist, end)
    for name, critic in [(f.__name__, f) for f in self.extra_critics]:
      terms[name] = critic(data)
    cost = torch.stack(list(terms.values())).sum(0)
    cost = torch.where(legal, cost, torch.inf)
    fail = ~legal.any(-1)
    best = cost.argmin(-1)
    rows = torch.arange(n, device=dev)
    cmd = torch.where(fail[:, None], 0.0, twist[rows, best])
    self._oscillation_sent(cmd)
    self.last_cmd = cmd
    self.plan = cmd[:, None].expand(-1, c.plan_steps, -1).clone()
    plan_v = accel_limited(self.plan[:, None], speed, c)
    plan_path = integrate(plan_v, pose0, c)[:, 0]
    weight = torch.zeros_like(cost)
    weight[rows, best] = (~fail).float()
    zeros = torch.zeros(n, c.plan_steps, device=dev)
    return {
      "cmd": cmd,
      "plan": self.plan,
      "path": plan_path,
      "pred_path": plan_path,
      "pred_std": torch.zeros(n, c.plan_steps, 3, device=dev),
      "p_fallen": torch.zeros(n, device=dev),
      "pred_p_fallen": zeros,
      "pred_p_collision": zeros,
      "p_fall_soon": torch.zeros(n, device=dev),
      "ess": torch.ones(n, device=dev),
      "cost_best": cost[rows, best],
      **{f"cost_{k}": t[rows, best] for k, t in terms.items()},
      **{f"spread_{k}": t.std(-1) for k, t in terms.items()},
      "nav2_fail": fail,
      "traj": traj,
      "cv": cv,
      "weight": weight,
      "cost": cost,
      "terms": terms,
    }

  # Critics (``dwb_critics``), each its scale times its raw score.
  def _critics(
    self, d: dict, twist: torch.Tensor, end: torch.Tensor
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    c = self.cfg
    traj, res = d["traj"], d["resolution"]
    last = traj[:, :, -1]  # [n, S, 3]
    terms: dict[str, torch.Tensor] = {}
    # BaseObstacle: the cost under the last pose (``sum_scores`` false); a
    # pose on an obstacle, the robot's inscribed band or unknown ground, or
    # a footprint point on an obstacle, rules the twist out.
    centre = _lookup(d["costmap"], traj[..., :2], res)
    edge = footprint_max_cost(d["costmap"], traj, c.costmap, res)
    legal = ~((centre >= INSCRIBED).any(-1) | (edge >= LETHAL).any(-1))
    terms["base_obstacle"] = c.base_obstacle_scale * centre[..., -1].where(
      centre[..., -1] < NO_INFORMATION, 0.0
    )
    # Path and goal distance (in cells) from the last pose; the align
    # critics from a point ``forward_point_distance`` ahead of it.
    ahead = last[..., :2] + c.forward_point_distance * torch.stack(
      [torch.cos(last[..., 2]), torch.sin(last[..., 2])], -1
    )
    to_goal = d["goal"].norm(dim=-1)[:, None]  # [n, 1]
    near = to_goal <= c.xy_goal_tolerance
    terms["path_dist"] = c.path_dist_scale * _nearest(last[..., :2], d["path"]) / res
    terms["goal_dist"] = c.goal_dist_scale * (last[..., :2] - end).norm(dim=-1) / res
    terms["path_align"] = c.path_align_scale * _nearest(ahead, d["path"]) / res * ~near
    terms["goal_align"] = c.goal_align_scale * (ahead - end).norm(dim=-1) / res
    # RotateToGoal: within the goal tolerance, only turning on the spot
    # (once the robot has stopped), toward the goal heading.
    planar = twist[..., :2].norm(dim=-1)
    legal = legal & ~(near & (planar > c.trans_stopped_velocity))
    yaw_err = _wrap(last[..., 2] - d["path_yaw"][:, None]).abs()
    terms["rotate_to_goal"] = c.rotate_to_goal_scale * yaw_err * near
    # Oscillation: an axis that has flipped may not flip back until the
    # robot has moved or turned far enough.
    back = self.flipped[:, None] & (torch.sign(twist) == -self.trend[:, None])
    legal = legal & ~(back & (twist.abs() > 1e-6)).any(-1)
    return terms, legal

  def _oscillation_moved(self, speed: torch.Tensor) -> None:
    c = self.cfg
    self.moved[:, 0] += speed[:, :2].norm(dim=-1) * c.model_dt
    self.moved[:, 1] += speed[:, 2].abs() * c.model_dt
    reset = (self.moved[:, 0] >= c.oscillation_reset_dist) | (
      self.moved[:, 1] >= c.oscillation_reset_angle
    )
    self.flipped &= ~reset[:, None]

  def _oscillation_sent(self, cmd: torch.Tensor) -> None:
    sign = torch.sign(torch.where(cmd.abs() > 1e-6, cmd, 0.0))
    flip = (sign != 0) & (self.trend != 0) & (sign != self.trend)
    self.moved[flip.any(-1)] = 0.0
    self.flipped |= flip
    self.trend = torch.where(sign != 0, sign, self.trend)
