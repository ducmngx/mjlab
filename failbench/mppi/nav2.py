"""Nav2's MPPI controller (``nav2_mppi_controller``), ported to batched torch.

The nominal local planner every FailBench arm shares. It follows Nav2's
optimizer and the critics of ``nav2_bringup/params/nav2_params.yaml``, with
their default weights; only what the Go2 needs differs (``Nav2MppiCfg``):
its speed limits and sampling noise, a 3 s horizon at the 10 Hz planning
cycle, and a motion model fitted to how the HIM policy tracks commands.

One cycle, per robot (``Optimizer::evalControl``):

1. predict the pose one cycle ahead at the measured speed (latency), and
   take the last sequence as the mean;
2. sample ``batch_size`` control sequences: the mean plus Gaussian noise
   (``vx_std``, ``vy_std``, ``wz_std``);
3. roll each out with the motion model: velocities follow the sampled
   commands within the acceleration limits, then integrate (omni);
4. score every trajectory with the critics (each adds ``weight * cost **
   power``) plus MPPI's control cost (``gamma``);
5. weight sequences by ``softmax(-(cost - min) / temperature)``, average,
   smooth (Savitzky-Golay), clamp to the limits; send the next command and
   shift the sequence as the next mean.

Maps are robot-centred and yaw-aligned (``failbench/sensing.cell_centres``),
so the robot sits at the origin facing +x. There is no global planner: the
path the path critics follow is the straight line from the robot to the
goal, pruned to ``prune_distance`` (Nav2's ``FeasiblePathHandler``), at the
costmap's resolution. Every FailBench scene keeps that line walkable.

The costmap is Nav2's: obstacles lethal (254), the robot's inscribed radius
253, then ``252 exp(-cost_scaling_factor (d - inscribed))`` out to
``inflation_radius``, from the obstacle cells of the height map.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from failbench.mppi.costmap import distance_field, obstacle_cells

LETHAL = 254.0
INSCRIBED = 253.0
NO_INFORMATION = 255.0


@dataclass(frozen=True)
class CriticCfg:
  enabled: bool = True
  weight: float = 1.0
  power: int = 1


@dataclass(frozen=True)
class CostCriticCfg(CriticCfg):
  weight: float = 3.81
  critical_cost: float = 300.0
  near_collision_cost: float = 253.0
  collision_cost: float = 1_000_000.0
  near_goal_distance: float = 1.0
  trajectory_point_step: int = 2
  consider_footprint: bool = True
  """Check the Go2's rectangle (``CostmapCfg.footprint``), not a circle."""


@dataclass(frozen=True)
class GoalCriticCfg(CriticCfg):
  weight: float = 5.0
  threshold_to_consider: float = 1.4


@dataclass(frozen=True)
class GoalAngleCriticCfg(CriticCfg):
  weight: float = 3.0
  threshold_to_consider: float = 0.5


@dataclass(frozen=True)
class PathAlignCriticCfg(CriticCfg):
  weight: float = 14.0
  max_path_occupancy_ratio: float = 0.05
  trajectory_point_step: int = 4
  threshold_to_consider: float = 0.5
  offset_from_furthest: int = 20


@dataclass(frozen=True)
class PathFollowCriticCfg(CriticCfg):
  weight: float = 5.0
  offset_from_furthest: int = 5
  threshold_to_consider: float = 1.4


@dataclass(frozen=True)
class PathAngleCriticCfg(CriticCfg):
  weight: float = 2.0
  offset_from_furthest: int = 4
  threshold_to_consider: float = 0.5
  max_angle_to_furthest: float = 1.0
  """Forward preference (mode 0): the robot may not reverse."""


@dataclass(frozen=True)
class PreferForwardCriticCfg(CriticCfg):
  weight: float = 5.0
  threshold_to_consider: float = 0.5


@dataclass(frozen=True)
class ConstraintCriticCfg(CriticCfg):
  weight: float = 4.0


@dataclass(frozen=True)
class TwirlingCriticCfg(CriticCfg):
  enabled: bool = False
  """Commented out in Nav2's bringup config."""
  weight: float = 10.0


@dataclass(frozen=True)
class CriticsCfg:
  constraint: ConstraintCriticCfg = field(default_factory=ConstraintCriticCfg)
  cost: CostCriticCfg = field(default_factory=CostCriticCfg)
  goal: GoalCriticCfg = field(default_factory=GoalCriticCfg)
  goal_angle: GoalAngleCriticCfg = field(default_factory=GoalAngleCriticCfg)
  path_align: PathAlignCriticCfg = field(default_factory=PathAlignCriticCfg)
  path_follow: PathFollowCriticCfg = field(default_factory=PathFollowCriticCfg)
  path_angle: PathAngleCriticCfg = field(default_factory=PathAngleCriticCfg)
  prefer_forward: PreferForwardCriticCfg = field(default_factory=PreferForwardCriticCfg)
  twirling: TwirlingCriticCfg = field(default_factory=TwirlingCriticCfg)


@dataclass(frozen=True)
class CostmapCfg:
  inflation_radius: float = 0.70
  cost_scaling_factor: float = 3.0
  footprint: tuple[float, float] = (0.70, 0.36)
  """The Go2's length and width (legs included), m."""
  footprint_points: int = 6
  """Points per side of the footprint rectangle checked for collision."""
  wall_jump_m: float = 0.18
  """Height step that makes a cell an obstacle (``obstacle_cells``)."""
  margin: float = 0.0
  """The inflation baseline: the highest cost short of the inscribed one
  (252) held ``margin`` m further out from obstacles, Nav2's decay beyond;
  0 is Nav2's layer. A soft keep-out: the cost critic charges it, the
  collision check doesn't."""
  severity_ref: float | None = None
  """Semantic inflation: each obstacle's margin is ``margin * severity /
  severity_ref`` (25, glass: crates 0.32x, walls 0.16x, people 2x); None: one
  margin for all."""


@dataclass(frozen=True)
class Nav2MppiCfg:
  batch_size: int = 1000
  time_steps: int = 30
  model_dt: float = 0.1
  """3 s at the 10 Hz planning cycle (Nav2: 56 x 0.05 s at 20 Hz)."""
  temperature: float = 0.3
  gamma: float = 0.015
  vx_max: float = 1.2
  vx_min: float = 0.0
  """The HIM policy's range; no walking backwards (Nav2: -0.35 to 0.5)."""
  vy_max: float = 0.5
  wz_max: float = 1.0
  ax_max: float = 3.0
  ax_min: float = -3.0
  ay_max: float = 3.0
  ay_min: float = -3.0
  az_max: float = 3.5
  vx_std: float = 0.4
  """Nav2's 0.2 is tuned for 0.5 m/s; the Go2 walks to 1.2 m/s."""
  vy_std: float = 0.2
  wz_std: float = 0.4
  vy_gain: float = 0.7
  """The Go2 walks about 0.71 of a commanded sidestep (measured on the
  course; forward and turning track within 3-6%)."""
  deadband: float = 0.2
  """The command term zeroes planar commands at or below this (m/s); the
  motion model walks them as standing."""
  prune_distance: float = 4.0
  """How far along the path the critics see (Nav2 scales it to the horizon
  at top speed: 3 s x 1.2 m/s)."""
  path_resolution: float = 0.05
  sgf_order: int = 2
  plan_steps: int = 50
  """Steps of the plan handed to the runner (``plan``): the sequence, then
  its last command held (the runner's logs and model use 5 s plans)."""
  offset_m: tuple[float, ...] = ()
  """Not Nav2: candidates that step the mean sequence sideways by each
  offset (m, +y left) at ``strafe_vy``, then carry on parallel to it, in
  place of as many noise samples (FailBench's own sampler's ``OFFSET_M``).
  Independent per-step noise rarely keeps a sustained sidestep."""
  strafe_vy: float = 0.25
  steady_candidates: tuple[tuple[float, float, float], ...] = ()
  """Not Nav2: constant (vx, vy, wz) commands held for the whole horizon, in
  place of the last noise samples (like DWB's velocity grid). Noise about
  the mean rarely holds one course for 3 s; these always offer it."""
  iteration_count: int = 1
  """Nav2's ``iteration_count``: optimisation passes per cycle, each sampling
  about the last pass's mean (Nav2's default 1)."""
  gap_quantile: float | None = None
  """Not Nav2: weigh samples by ``softmax(-(cost - min) / (temperature *
  (q - min)))``, ``q`` this cost quantile, as FailBench's own planner, so
  the temperature is relative to the batch's cost gap. None: Nav2's
  absolute ``temperature``."""
  log_normal_std: float = 0.0
  """Not Nav2: log-MPPI's sampling (Mohamed, Allibert, Martin, "Autonomous
  navigation of AGVs in unknown cluttered environments: log-MPPI control
  strategy", RA-L 2022): each noise value is the Gaussian one times a
  log-normal factor ``exp(z)``, ``z ~ N(-s^2, s^2)`` with ``s`` this, so each
  axis keeps its spread (E[exp(2z)] = 1) but far-out samples are more common
  (kurtosis ``3 exp(4 s^2)``). 0: Nav2's Gaussian."""
  noise_corr: tuple[float, float, float] = (0.0, 0.0, 0.0)
  """Not Nav2: each axis's (vx, vy, wz) noise is an AR(1) process along the
  horizon with this step-to-step correlation, each step keeping its std.
  Independent noise averages a sample's sideways speed out over 30 steps
  (sd ``vy_std / sqrt(30)``); correlated noise keeps a sample drifting one
  way, so the batch holds sustained sidesteps at full speed. 0: Nav2's."""
  critics: CriticsCfg = field(default_factory=CriticsCfg)
  costmap: CostmapCfg = field(default_factory=CostmapCfg)


def inflate(
  obst: torch.Tensor,
  resolution: float,
  cfg: CostmapCfg,
  severity: torch.Tensor | None = None,
) -> torch.Tensor:
  """Nav2's inflation layer: cost per cell ``[n, H, W]`` from an obstacle
  mask, the inscribed radius from the footprint's half width; pushed out by
  ``cfg.margin``, per obstacle by its ``severity [n, H, W]`` when
  ``cfg.severity_ref`` is set."""
  inscribed = cfg.footprint[1] / 2
  levels = [(cfg.margin, obst)]
  if cfg.margin > 0 and cfg.severity_ref is not None and severity is not None:
    levels = [
      (cfg.margin * v / cfg.severity_ref, obst & (severity == v))
      for v in severity[obst].unique().tolist()
    ]
  cost = torch.zeros_like(obst, dtype=torch.float32)
  for margin, cells in levels:
    reach = cfg.inflation_radius + margin
    dist = distance_field(cells, resolution, reach + resolution)
    decay = (INSCRIBED - 1) * torch.exp(
      -cfg.cost_scaling_factor * (dist - inscribed - margin).clamp(min=0)
    )
    cost = torch.maximum(cost, torch.where(dist <= reach, decay.floor(), 0.0))
  dist = distance_field(obst, resolution, inscribed + resolution)
  cost = torch.where(dist <= inscribed, INSCRIBED, cost)
  return torch.where(obst, LETHAL, cost)


def _lookup(costmap: torch.Tensor, xy: torch.Tensor, resolution: float) -> torch.Tensor:
  """Cost at metric points ``xy [n, ..., 2]`` (nearest cell; off the map:
  no information)."""
  n, h, w = costmap.shape
  col = torch.round(xy[..., 0] / resolution + (w - 1) / 2).long()
  row = torch.round(xy[..., 1] / resolution + (h - 1) / 2).long()
  inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
  flat = (row.clamp(0, h - 1) * w + col.clamp(0, w - 1)).view(n, -1)
  cost = costmap.view(n, -1).gather(1, flat).view(col.shape)
  return torch.where(inside, cost, NO_INFORMATION)


def _wrap(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


def _sg_filter(seq: torch.Tensor, hist: torch.Tensor, order: int) -> torch.Tensor:
  """Nav2's Savitzky-Golay smoothing of ``seq [n, T, 3]`` over a 9-point
  window, the 4 past commands ``hist [n, 4, 3]`` before it and the last
  point repeated after."""
  if order == 1:
    taps = torch.full((9,), 1 / 9, device=seq.device)
  else:
    taps = torch.tensor(
      [-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0], device=seq.device
    )
    taps = taps / 231.0
  steps = seq.shape[1] - 1
  if steps < 20:
    return seq
  padded = torch.cat([hist, seq[:, : steps + 1], seq[:, -1:].expand(-1, 4, -1)], 1)
  windows = padded.unfold(1, 9, 1)[:, :steps]  # [n, steps, 3, 9]
  out = seq.clone()
  out[:, :steps] = (windows * taps).sum(-1)
  return out


def accel_limited(cv: torch.Tensor, v0: torch.Tensor, c: Any) -> torch.Tensor:
  """Velocities ``[n, K, T, 3]`` the robot reaches following sampled
  commands ``cv`` from ``v0 [n, 3]``, within the acceleration limits
  (``MotionModel::predict``): step i follows command i - 1. ``c``: a
  config with the acceleration limits and ``model_dt`` (``Nav2MppiCfg``,
  ``DwbCfg``)."""
  dt = c.model_dt
  lo = torch.tensor([c.ax_min, c.ay_min, -c.az_max], device=cv.device) * dt
  hi = torch.tensor([c.ax_max, c.ay_max, c.az_max], device=cv.device) * dt
  v = [v0[:, None].expand(-1, cv.shape[1], -1)]
  for i in range(1, cv.shape[2]):
    prev = v[-1]
    # vx, vy: accelerating away from zero is ax_max, braking ax_min.
    pos = prev >= 0
    low = torch.where(pos, prev + lo, prev - hi)
    high = torch.where(pos, prev + hi, prev - lo)
    low[..., 2], high[..., 2] = prev[..., 2] + lo[2], prev[..., 2] + hi[2]
    v.append(torch.minimum(torch.maximum(cv[:, :, i - 1], low), high))
  return torch.stack(v, 2)


def integrate(v: torch.Tensor, pose0: torch.Tensor, c: Any) -> torch.Tensor:
  """Poses ``[n, K, T, 3]`` from velocities ``v [n, K, T, 3]`` and a start
  pose ``pose0 [n, 3]`` (omni; the Go2's sidestep gain and deadband from
  ``c``)."""
  dt = c.model_dt
  planar = v[..., :2].norm(dim=-1, keepdim=True)
  walk = torch.where(planar <= c.deadband, 0.0, v[..., :2])
  vx, vy = walk[..., 0], c.vy_gain * walk[..., 1]
  yaw = pose0[:, None, None, 2] + torch.cumsum(v[..., 2] * dt, -1)
  # Each step translates along the heading at its start.
  head = torch.cat([pose0[:, None, None, 2].expand_as(yaw[..., :1]), yaw[..., :-1]], -1)
  cy, sy = torch.cos(head), torch.sin(head)
  x = pose0[:, None, None, 0] + torch.cumsum((vx * cy - vy * sy) * dt, -1)
  y = pose0[:, None, None, 1] + torch.cumsum((vx * sy + vy * cy) * dt, -1)
  return torch.stack([x, y, yaw], -1)


def _outline(cm: CostmapCfg, device) -> torch.Tensor:
  """The footprint rectangle's outline points ``[F, 2]`` in the body frame."""
  hl, hw = cm.footprint[0] / 2, cm.footprint[1] / 2
  t = torch.linspace(-1, 1, cm.footprint_points, device=device)
  return torch.cat(
    [
      torch.stack([t * hl, torch.full_like(t, hw)], -1),
      torch.stack([t * hl, torch.full_like(t, -hw)], -1),
      torch.stack([torch.full_like(t, hl), t * hw], -1),
      torch.stack([torch.full_like(t, -hl), t * hw], -1),
    ]
  )


def footprint_points(pts: torch.Tensor, cm: CostmapCfg) -> torch.Tensor:
  """Points on the footprint rectangle's outline at poses ``[..., 3]``."""
  local = _outline(cm, pts.device)
  lx, ly = local[:, 0], local[:, 1]
  cy, sy = torch.cos(pts[..., 2:3]), torch.sin(pts[..., 2:3])
  x = pts[..., 0:1] + cy * lx - sy * ly
  y = pts[..., 1:2] + sy * lx + cy * ly
  return torch.stack([x, y], -1)


def footprint_max_cost(
  costmap: torch.Tensor, pts: torch.Tensor, cm: CostmapCfg, resolution: float
) -> torch.Tensor:
  """The highest cost under the footprint outline at poses ``[n, ..., 3]``:
  ``_lookup`` of ``footprint_points`` maxed over the outline, one outline
  point at a time so a whole batch of rollouts never holds every point (at
  4096 robots that tensor alone is tens of GB)."""
  cy, sy = torch.cos(pts[..., 2]), torch.sin(pts[..., 2])
  out = None
  for lx, ly in _outline(cm, pts.device).tolist():
    x = pts[..., 0] + cy * lx - sy * ly
    y = pts[..., 1] + sy * lx + cy * ly
    cost = _lookup(costmap, torch.stack([x, y], -1), resolution)
    out = cost if out is None else torch.maximum(out, cost)
  assert out is not None
  return out


def straight_path(start: torch.Tensor, goal: torch.Tensor, c: Any):
  """The straight line from ``start [n, 2]`` to ``goal [n, 2]``, pruned
  to ``prune_distance``: points ``[n, P, 2]``, their yaw ``[n]``, how many
  are real ``[n]`` (the rest repeat the last) and its length ``[n]``; ``c``
  holds ``prune_distance`` and ``path_resolution``."""
  count = round(c.prune_distance / c.path_resolution) + 1
  d = goal - start
  length = d.norm(dim=-1).clamp(max=c.prune_distance)
  yaw = torch.atan2(d[:, 1], d[:, 0])
  s = torch.arange(count, device=start.device) * c.path_resolution
  real = (length / c.path_resolution).floor().long() + 1
  s = torch.minimum(s[None], length[:, None])
  unit = torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
  pts = start[:, None] + s[..., None] * unit[:, None]
  return pts, yaw, real, length


class Nav2Mppi:
  """Batched Nav2 MPPI for ``n`` robots. ``extra_critics``: callables
  ``(data) -> cost [n, K]`` added to the critics' sum (e.g. the fault risk);
  ``data`` holds the trajectories, controls, path, goal and maps."""

  CRITICS = (
    "constraint",
    "cost",
    "goal",
    "goal_angle",
    "path_align",
    "path_follow",
    "path_angle",
    "prefer_forward",
    "twirling",
  )
  """The stock critics' names (their ``step`` cost terms)."""

  def __init__(self, cfg: Nav2MppiCfg, n: int, device, extra_critics=()):
    self.cfg = cfg
    self.n = n
    self.device = device
    self.extra_critics = list(extra_critics)
    self.critic_scale: dict[str, torch.Tensor] = {}
    """Per-robot factors ``[n]`` on named critics (the planner-driven data's
    driving styles, ``failbench/driver.py``); empty is stock Nav2."""
    self.temperature_scale = torch.ones(n, device=device)
    """Per-robot factor on ``temperature`` (1 is stock)."""
    t = cfg.time_steps
    self.seq = torch.zeros(n, t, 3, device=device)
    """The control sequence (vx, vy, wz); ``seq[:, 1]`` is sent next."""
    self.hist = torch.zeros(n, 4, 3, device=device)
    self.last_cmd = torch.zeros(n, 3, device=device)
    self.substeps = 5
    """Control steps per cycle (50 Hz policy, 10 Hz planning)."""
    self.plan = torch.zeros(n, cfg.plan_steps, 3, device=device)
    self.still_s = torch.zeros(n, device=device)

  # The runner's planner interface.
  def reset(self, ids: torch.Tensor, cmd: torch.Tensor | None = None) -> None:
    self.seq[ids] = 0.0 if cmd is None else cmd[:, None]
    self.hist[ids] = 0.0 if cmd is None else cmd[:, None]
    self.last_cmd[ids] = 0.0 if cmd is None else cmd
    self.plan[ids] = 0.0 if cmd is None else cmd[:, None]

  def track(self, pos_xy: torch.Tensor, dt: float) -> None:
    del pos_xy, dt

  def advance(self) -> torch.Tensor:
    self.plan = torch.cat([self.plan[:, 1:], self.plan[:, -1:]], dim=1)
    return self.plan[:, 0]

  def branch_names(self, num_joints: int = 12) -> list[str]:
    del num_joints
    return []

  # Motion model.
  def _predict(self, cv: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    return accel_limited(cv, v0, self.cfg)

  def _integrate(self, v: torch.Tensor, pose0: torch.Tensor) -> torch.Tensor:
    return integrate(v, pose0, self.cfg)

  # Path.
  def _path(self, start: torch.Tensor, goal: torch.Tensor):
    return straight_path(start, goal, self.cfg)

  @torch.no_grad()
  def step(
    self,
    costmap: torch.Tensor,
    resolution: float,
    goal: torch.Tensor,
    speed: torch.Tensor,
    context: dict[str, Any] | None = None,
    path: tuple[torch.Tensor, ...] | None = None,
  ) -> dict[str, Any]:
    """Plan one cycle. ``costmap [n, H, W]`` (``inflate``), ``goal [n, 2]``
    and ``speed [n, 3]`` (measured vx, vy, wz) in the robot's yaw frame.
    ``context`` is passed to the extra critics. ``path``: the global path to
    track, as ``straight_path`` gives it (default: straight at the goal).
    Returns ``cmd [n, 3]``, the new ``plan``, its kinematic ``path``, the
    trajectories' ``cost`` and ``weight``, and each critic's cost on the
    heaviest trajectory."""
    c = self.cfg
    n, dev = self.n, self.device
    dt = c.model_dt
    # Measured speed, moved toward the last command within one cycle's
    # acceleration (``Optimizer::prepare``), and the pose one cycle on.
    lo = torch.tensor([c.ax_min, c.ay_min, -c.az_max], device=dev) * dt
    hi = torch.tensor([c.ax_max, c.ay_max, c.az_max], device=dev) * dt
    v0 = self.last_cmd.clamp(speed + lo, speed + hi)
    pose0 = torch.zeros(n, 3, device=dev)
    pose0[:, 0] = speed[:, 0] * dt
    pose0[:, 1] = c.vy_gain * speed[:, 1] * dt
    pose0[:, 2] = speed[:, 2] * dt
    # The mean, pinned to the current speed at step 0, within limits.
    self.seq = self._constrain(self.seq, v0)
    # Nav2's ``iteration_count``: sample, score and average again about the
    # new mean before sending anything.
    args = (costmap, resolution, goal, v0, pose0, context, path)
    mean, weight, cost, terms, fail, traj, cv = self._iterate(*args)
    for _ in range(c.iteration_count - 1):
      mean, weight, cost, terms, fail, traj, cv = self._iterate(*args)
    cmd = mean[:, 1]
    self.hist = torch.cat([self.hist[:, 1:], cmd[:, None]], 1)
    self.last_cmd = cmd
    # The plan the runner plays out: the sequence from the command sent on,
    # its last command held to ``plan_steps``.
    rest = mean[:, 1:]
    hold = rest[:, -1:].expand(-1, c.plan_steps - rest.shape[1], -1)
    self.plan = torch.cat([rest, hold], 1)
    plan_v = self._predict(self.plan[:, None], v0)[:, 0]
    plan_path = self._integrate(plan_v[:, None], pose0)[:, 0]
    # Shift the sequence for next cycle's mean.
    self.seq = torch.cat([mean[:, 1:], mean[:, -1:]], 1)
    best = weight.argmax(-1)
    rows = torch.arange(n, device=dev)
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
      "ess": 1.0 / weight.square().sum(-1),
      "cost_best": cost[rows, best],
      **{f"cost_{k}": t[rows, best] for k, t in terms.items()},
      # How much each critic tells the samples apart (the softmax sees only
      # differences, at ``temperature``).
      **{f"spread_{k}": t.std(-1) for k, t in terms.items()},
      "nav2_fail": fail,
      "traj": traj,
      "cv": cv,
      "weight": weight,
      "cost": cost,
      "terms": terms,
    }

  def _iterate(self, costmap, resolution, goal, v0, pose0, context, path):
    """One optimisation pass about ``self.seq``: sample, roll out, score,
    average; leaves the new mean in ``self.seq`` and returns it with the
    pass's ``weight``, ``cost``, ``terms``, ``fail``, ``traj`` and ``cv``."""
    c, dev = self.cfg, self.device
    std = torch.tensor([c.vx_std, c.vy_std, c.wz_std], device=dev)
    cv = self.seq[:, None] + self._noise()  # [n, K, T, 3]
    if c.offset_m:
      cv[:, -len(c.offset_m) :] = self._offsets()
    if c.steady_candidates:
      steady = torch.tensor(c.steady_candidates, device=dev)
      cv[:, -len(steady) :] = steady[:, None]
    v = self._predict(cv, v0)
    traj = self._integrate(v, pose0)
    track = path if path is not None else self._path(pose0[:, :2], goal)
    points, path_yaw, path_real, path_len = track
    data = {
      "traj": traj,
      "v": v,
      "cv": cv,
      "v0": v0,
      "commands": self._sent(cv, v0) if self.extra_critics else None,
      "pose0": pose0,
      "path": points,
      "path_yaw": path_yaw,
      "path_real": path_real,
      "path_length": path_len,
      "goal": goal,
      "costmap": costmap,
      "resolution": resolution,
      **(context or {}),
    }
    terms, fail = self._critics(data)
    data["stock_cost"] = torch.stack(list(terms.values())).sum(0)
    for name, critic in [(f.__name__, f) for f in self.extra_critics]:
      terms[name] = critic(data)
    for name, scale in self.critic_scale.items():
      if name in terms:
        terms[name] = terms[name] * scale[:, None]
    cost = torch.stack(list(terms.values())).sum(0)
    # MPPI's control cost: noise against the mean, per axis.
    gamma = c.gamma / std.square()
    cost = cost + ((cv - self.seq[:, None]) * self.seq[:, None] * gamma).sum((-1, -2))
    low = cost.amin(-1, keepdim=True)
    temperature = c.temperature * self.temperature_scale[:, None]
    scale = temperature
    if c.gap_quantile is not None:
      gap = cost.quantile(c.gap_quantile, dim=-1, keepdim=True) - low
      scale = temperature * gap + 1e-6
    weight = torch.softmax(-(cost - low) / scale, -1)
    mean = (weight[..., None, None] * cv).sum(1)
    mean = _sg_filter(mean, self.hist, c.sgf_order)
    mean = self._constrain(mean, v0)
    # Nav2 resets when every trajectory collides; the robot stops.
    mean = torch.where(fail[:, None, None], 0.0, mean)
    self.seq = mean
    return mean, weight, cost, terms, fail, traj, cv

  def _noise(self) -> torch.Tensor:
    """Sampling noise ``[n, K, T, 3]``: Gaussian at ``vx_std``, ``vy_std``,
    ``wz_std``, or log-MPPI's normal times log-normal (``log_normal_std``),
    correlated along the horizon by ``noise_corr``."""
    c = self.cfg
    std = torch.tensor([c.vx_std, c.vy_std, c.wz_std], device=self.device)
    shape = (self.n, c.batch_size, c.time_steps, 3)
    noise = torch.randn(shape, device=self.device) * std
    s = c.log_normal_std
    if s > 0:
      noise = noise * torch.exp(torch.randn(shape, device=self.device) * s - s * s)
    if any(c.noise_corr):
      rho = torch.tensor(c.noise_corr, device=self.device)
      keep = (1 - rho * rho).sqrt()
      steps = [noise[:, :, 0]]
      for t in range(1, c.time_steps):
        steps.append(rho * steps[-1] + keep * noise[:, :, t])
      noise = torch.stack(steps, 2)
    return noise

  def _offsets(self) -> torch.Tensor:
    """The mean sequence stepped sideways by each ``offset_m`` ``[n, O, T,
    3]``: strafing at ``strafe_vy`` until the motion model has walked the
    offset (``vy_gain`` of the command)."""
    c = self.cfg
    t = (torch.arange(c.time_steps, device=self.device) + 0.5) * c.model_dt
    out = []
    for off in c.offset_m:
      walk_s = abs(off) / (c.strafe_vy * c.vy_gain)
      vy = math.copysign(c.strafe_vy, off) * (t < walk_s)
      shift = torch.zeros(c.time_steps, 3, device=self.device)
      shift[:, 1] = vy
      out.append(self.seq + shift)
    return torch.stack(out, 1)

  def _sent(self, cv: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """What the robot would be sent if it committed to each sample ``cv [n,
    K, T, 3]``: within the limits, from the command sent next, the last held
    (as the planner-driven data commits, ``PlannerDriver.commit``)."""
    n, k = cv.shape[:2]
    sent = self._constrain(cv.flatten(0, 1), v0.repeat_interleave(k, 0))
    return torch.cat([sent[:, 1:], sent[:, -1:]], 1).unflatten(0, (n, k))

  def _constrain(self, seq: torch.Tensor, v0: torch.Tensor) -> torch.Tensor:
    """Clamp a sequence ``[n, T, 3]`` to the speed limits and, from the
    current speed ``v0`` at step 0, the acceleration limits
    (``applyControlSequenceConstraints``, shifting on)."""
    c = self.cfg
    vmin = torch.tensor([c.vx_min, -c.vy_max, -c.wz_max], device=seq.device)
    vmax = torch.tensor([c.vx_max, c.vy_max, c.wz_max], device=seq.device)
    dlo = torch.tensor([c.ax_min, c.ay_min, -c.az_max], device=seq.device) * c.model_dt
    dhi = torch.tensor([c.ax_max, c.ay_max, c.az_max], device=seq.device) * c.model_dt
    out = [v0]
    for i in range(1, seq.shape[1]):
      x = torch.minimum(torch.maximum(seq[:, i], vmin), vmax)
      out.append(torch.minimum(torch.maximum(x, out[-1] + dlo), out[-1] + dhi))
    return torch.stack(out, 1)

  # Critics (``nav2_mppi_controller/src/critics``).
  def _critics(self, d: dict) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    cc = self.cfg.critics
    terms: dict[str, torch.Tensor] = {}
    fail = torch.zeros(self.n, dtype=torch.bool, device=self.device)
    furthest = self._furthest(d)
    valid = self._path_valid(d)
    if cc.constraint.enabled:
      terms["constraint"] = self._weigh(cc.constraint, self._constraint(d))
    if cc.cost.enabled:
      cost, fail = self._cost(d)
      terms["cost"] = self._weigh(cc.cost, cost)
    if cc.goal.enabled:
      terms["goal"] = self._weigh(cc.goal, self._goal(d))
    if cc.goal_angle.enabled:
      terms["goal_angle"] = self._weigh(cc.goal_angle, self._goal_angle(d))
    if cc.path_align.enabled:
      terms["path_align"] = self._weigh(
        cc.path_align, self._path_align(d, furthest, valid)
      )
    if cc.path_follow.enabled:
      terms["path_follow"] = self._weigh(
        cc.path_follow, self._path_follow(d, furthest, valid)
      )
    if cc.path_angle.enabled:
      terms["path_angle"] = self._weigh(cc.path_angle, self._path_angle(d, furthest))
    if cc.prefer_forward.enabled:
      terms["prefer_forward"] = self._weigh(cc.prefer_forward, self._prefer_forward(d))
    if cc.twirling.enabled:
      terms["twirling"] = self._weigh(cc.twirling, d["v"][..., 2].abs().mean(-1))
    return terms, fail

  @staticmethod
  def _weigh(cfg: CriticCfg, cost: torch.Tensor) -> torch.Tensor:
    return (cost * cfg.weight) ** cfg.power if cfg.power > 1 else cost * cfg.weight

  def _active(self, d: dict, below: bool, threshold: float) -> torch.Tensor:
    """``[n, 1]``: the path left is below (or at least) ``threshold``."""
    length = d["path_length"][:, None]
    return (length <= threshold) if below else (length >= threshold)

  def _constraint(self, d: dict) -> torch.Tensor:
    c = self.cfg
    vx, vy = d["v"][..., 0], d["v"][..., 1]
    over = (vx - c.vx_max).clamp(min=0) + (c.vx_min - vx).clamp(min=0)
    over = over + (vy.abs() - c.vy_max).clamp(min=0)
    return (over * c.model_dt).sum(-1)

  def _cost(self, d: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """CostCritic: the costmap under each trajectory's points (every
    ``trajectory_point_step``), a footprint check where the centre is close
    enough for any part to touch, a collision ending the trajectory at
    ``collision_cost``."""
    cfg, cm = self.cfg.critics.cost, self.cfg.costmap
    res, costmap = d["resolution"], d["costmap"]
    pts = d["traj"][:, :, :: cfg.trajectory_point_step]  # [n, K, S, 3]
    centre = _lookup(costmap, pts[..., :2], res)
    if cfg.consider_footprint:
      edge = footprint_max_cost(costmap, pts, cm, res)
      circum = math.hypot(cm.footprint[0] / 2, cm.footprint[1] / 2)
      inscribed = cm.footprint[1] / 2
      possible = (INSCRIBED - 1) * math.exp(
        -cm.cost_scaling_factor * (circum - inscribed)
      )
      check = centre >= possible
      score = torch.where(check, torch.maximum(edge, centre), centre)
      hit = (score >= LETHAL) & (score < NO_INFORMATION)
    else:
      hit = (centre >= INSCRIBED) & (centre < NO_INFORMATION)
    hit = hit | (centre >= NO_INFORMATION)
    near_goal = d["path_length"][:, None, None] < cfg.near_goal_distance
    step_cost = torch.where(
      centre >= cfg.near_collision_cost,
      cfg.critical_cost,
      torch.where(near_goal, 0.0, centre),
    )
    step_cost = torch.where(centre < 1.0, 0.0, step_cost)
    # Points from the first collision on don't add up; a collision sets the
    # trajectory's cost outright.
    before = torch.cumsum(hit.int(), -1) == 0
    total = (step_cost * before).sum(-1)
    collide = hit.any(-1)
    total = torch.where(collide, cfg.collision_cost, total)
    fail = collide.all(-1)
    return total / (254.0 * pts.shape[2]), fail

  def _goal(self, d: dict) -> torch.Tensor:
    cfg = self.cfg.critics.goal
    end = d["path"].gather(1, (d["path_real"] - 1)[:, None, None].expand(-1, 1, 2))[
      :, 0
    ]
    dist = (d["traj"][..., :2] - end[:, None, None]).norm(dim=-1).mean(-1)
    return dist * self._active(d, True, cfg.threshold_to_consider)

  def _goal_angle(self, d: dict) -> torch.Tensor:
    cfg = self.cfg.critics.goal_angle
    err = _wrap(d["traj"][..., 2] - d["path_yaw"][:, None, None]).abs().mean(-1)
    return err * self._active(d, True, cfg.threshold_to_consider)

  def _furthest(self, d: dict) -> torch.Tensor:
    """``findPathFurthestReachedPoint``: the furthest path index ``[n]``
    any trajectory's end is nearest to, among the points it could reach."""
    traj, path = d["traj"][..., :2], d["path"]
    seg = (traj[:, :, 1:] - traj[:, :, :-1]).norm(dim=-1).sum(-1)  # [n, K]
    s = torch.arange(path.shape[1], device=path.device) * self.cfg.path_resolution
    reach = (s[None, None] < seg[..., None]).sum(-1).clamp(max=path.shape[1] - 1)
    reach = torch.minimum(reach, (d["path_real"] - 1)[:, None])
    end = traj[:, :, -1]
    dist = (path[:, None] - end[:, :, None]).norm(dim=-1)  # [n, K, P]
    idx = torch.arange(path.shape[1], device=path.device)
    dist = torch.where(idx[None, None] <= reach[..., None], dist, torch.inf)
    return dist.argmin(-1).amax(-1)

  def _path_valid(self, d: dict) -> torch.Tensor:
    """Path points off obstacles and their inscribed band ``[n, P]``."""
    cost = _lookup(d["costmap"], d["path"], d["resolution"])
    return cost < INSCRIBED

  def _path_align(
    self, d: dict, furthest: torch.Tensor, valid: torch.Tensor
  ) -> torch.Tensor:
    cfg = self.cfg.critics.path_align
    traj, path = d["traj"], d["path"]
    n, k = traj.shape[:2]
    pts = traj[:, :, :: cfg.trajectory_point_step, :2]
    on = self._active(d, False, cfg.threshold_to_consider)[:, 0]
    on = on & (furthest >= cfg.offset_from_furthest)
    # Too much of the path reached is blocked: the path critics stand down.
    idx = torch.arange(path.shape[1], device=path.device)
    seen = idx[None] < furthest[:, None]
    blocked = (~valid & seen).sum(-1).float()
    on = on & ~(
      (blocked / furthest.clamp(min=1) > cfg.max_path_occupancy_ratio) & (blocked > 2)
    )
    # Each sampled point against the path point as far along the path as
    # the trajectory has walked (``findClosestPathPt``).
    walked = torch.cumsum(
      torch.cat(
        [
          torch.zeros(n, k, 1, device=traj.device),
          (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1),
        ],
        -1,
      ),
      -1,
    )[:, :, 1:]
    near = torch.round(walked / self.cfg.path_resolution).long()
    near = torch.minimum(near, (furthest - 1).clamp(min=0)[:, None, None])
    target = path.gather(1, near.view(n, -1, 1).expand(-1, -1, 2)).view(*near.shape, 2)
    ok = valid.gather(1, near.view(n, -1)).view(near.shape).float()
    dist = (target - pts[:, :, 1:]).norm(dim=-1)
    cost = (dist * ok).sum(-1) / ok.sum(-1).clamp(min=1)
    return cost * on[:, None]

  def _path_follow(
    self, d: dict, furthest: torch.Tensor, valid: torch.Tensor
  ) -> torch.Tensor:
    cfg = self.cfg.critics.path_follow
    path = d["path"]
    last = d["path_real"] - 1
    on = self._active(d, False, cfg.threshold_to_consider)[:, 0] & (last >= 1)
    idx = torch.minimum(furthest + cfg.offset_from_furthest, last)
    # On to the first valid point, stopping one short of the end.
    p = torch.arange(path.shape[1], device=path.device)[None]
    cand = (p >= idx[:, None]) & (p < (last - 1)[:, None]) & valid
    first = torch.where(cand, p, path.shape[1]).amin(-1)
    idx = torch.where(first < path.shape[1], first, torch.maximum(idx, last - 1))
    idx = torch.minimum(idx, last)
    target = path.gather(1, idx[:, None, None].expand(-1, 1, 2))[:, 0]
    dist = (d["traj"][:, :, -1, :2] - target[:, None]).norm(dim=-1)
    return dist * on[:, None]

  def _path_angle(self, d: dict, furthest: torch.Tensor) -> torch.Tensor:
    cfg = self.cfg.critics.path_angle
    path = d["path"]
    idx = torch.minimum(furthest + cfg.offset_from_furthest, d["path_real"] - 1)
    target = path.gather(1, idx[:, None, None].expand(-1, 1, 2))[:, 0]
    pose0 = d["pose0"]
    to = target - pose0[:, :2]
    off = _wrap(torch.atan2(to[:, 1], to[:, 0]) - pose0[:, 2]).abs()
    on = self._active(d, False, cfg.threshold_to_consider)[:, 0]
    on = on & (off >= cfg.max_angle_to_furthest)
    end = d["traj"][:, :, -1]
    rel = target[:, None] - end[..., :2]
    err = _wrap(torch.atan2(rel[..., 1], rel[..., 0]) - end[..., 2]).abs()
    return err * on[:, None]

  def _prefer_forward(self, d: dict) -> torch.Tensor:
    cfg = self.cfg.critics.prefer_forward
    back = (-d["v"][..., 0]).clamp(min=0).sum(-1) * self.cfg.model_dt
    return back * self._active(d, False, cfg.threshold_to_consider)


def nav2_costmap(
  height: torch.Tensor,
  resolution: float,
  cfg: CostmapCfg,
  severity: torch.Tensor | None = None,
) -> torch.Tensor:
  """The Nav2 costmap of a planning height map ``[n, H, W]`` (NaN unknown,
  counted as free: the oracle map knows every cell), with each cell's
  ``severity`` for semantic inflation."""
  obst = obstacle_cells(height, cfg.wall_jump_m)
  return inflate(obst, resolution, cfg, severity)
