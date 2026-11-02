"""A stock planner as the data's driver (the planner-driven data mode).

Before the fault a stock planner (Nav2 MPPI, ``failbench/mppi/nav2.py``, or
DWB, ``failbench/mppi/dwb.py``) drives each trial toward its goal, closed
loop, around its phantom obstacles (``failbench/phantoms.py``: in its
costmap only), so the state and history at the fault are a planner's: it
sidesteps, turns, slows and stops. At the fault the trial commits to one of
that cycle's candidates (``commit``), which then runs open loop for
``commit_steps``: the commands the impact model is given, and the robot's
path under them its target. Most commits are a random candidate of the
batch, which is what the fault-risk critic scores; the rest are the
planner's own plan.

So that the movements before and at the fault are not all straight runs
at a goal straight ahead, the planner can instead track a global plan, as a
stock local planner tracks its global planner's: a stretch of a path from a
``library`` planned on phantom maps (``failbench/scripts/plan_paths.py``),
the map laid down so the stretch starts at the robot, facing along it, and
in its costmap instead of the phantoms. A trial can also drive in its own
style (each critic's weight, and MPPI's temperature, scaled by a random
factor) and commit to random steps of commands instead of a candidate
(``random_prob``); a trial stuck behind its phantoms can lose them
(``unblock_s``). All are off by default (the stock planner straight at the
goal, around its phantoms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch

from failbench.maps import rasterize
from failbench.mppi.dwb import Dwb, DwbCfg
from failbench.mppi.nav2 import Nav2Mppi, Nav2MppiCfg, inflate, nav2_costmap
from failbench.paths import local_path
from failbench.phantoms import PhantomCfg, phantom_height, sample_phantoms
from failbench.sensing import cell_centres

MAP_RESOLUTION = 0.05

SOURCES = (
  "mppi_sample",
  "mppi_plan",
  "dwb_twist",
  "dwb_best",
  "mppi_random",
  "dwb_random",
)
"""What a trial committed to at its fault (``PlannerDriver.source``)."""
STYLE_COLUMNS = 10
"""``PlannerDriver.style`` columns: the planner's ``CRITICS`` factors in
order, then MPPI's temperature factor, the rest 1."""


@dataclass(frozen=True)
class PlannerDriveCfg:
  planner: Literal["nav2", "dwb"] = "nav2"
  nav2: Nav2MppiCfg = field(default_factory=lambda: Nav2MppiCfg(batch_size=256))
  """Nav2 MPPI's defaults but a smaller batch, for thousands of trials."""
  dwb: DwbCfg = field(default_factory=DwbCfg)
  phantoms: PhantomCfg = field(default_factory=PhantomCfg)
  map_cells: int = 120
  """The planner's costmap side (cells of ``MAP_RESOLUTION``): 6 m."""
  sample_prob: float = 0.8
  """Share of commits that take a uniformly random candidate of the batch;
  the rest take the planner's own plan."""
  random_prob: float = 0.0
  """Share of commits that are instead random steps of commands within the
  planner's limits (``random_hold_s``): movements the planner seldom picks
  (hard turns, sidesteps, stop and go). MPPI's are ramped as its samples
  are sent; DWB's step, as its twists do, and never sidestep."""
  random_hold_s: tuple[int, int] = (5, 15)
  """How many planning steps (0.1 s) each random command holds."""
  unblock_s: float | None = None
  """A walking trial that has not moved for this long loses its phantoms,
  as if they had moved away, so a planner that cannot get round them (DWB
  never sidesteps) sets off again instead of standing until its fault.
  None: never."""
  commit_steps: int = 30
  """Planning steps (0.1 s) a commit runs open loop: 3 s."""
  style_range: tuple[float, float] = (1.0, 1.0)
  """Each trial scales every critic's weight (and MPPI's temperature) by
  its own factor, log-uniform in this range; (1, 1) is stock."""
  library: str | None = None
  """A path library (``failbench/scripts/plan_paths.py``): each trial
  tracks a stretch of one of its paths, on its map; None: straight at the
  goal, around phantoms."""
  segment_m: tuple[float, float] = (3.0, 8.0)
  """How long a trial's stretch of library path is (uniform; shorter where
  the path ends first)."""
  corner_prob: float = 0.0
  """Share of library trials whose stretch starts ``corner_ahead_m`` before
  a corner of its path (``corners``), so the robot turns about when its
  fault fires; the rest start anywhere."""
  corner_ahead_m: tuple[float, float] = (1.0, 3.5)


def corners(
  paths: torch.Tensor,
  real: torch.Tensor,
  resolution: float,
  min_deg: float = 30.0,
  half_m: float = 0.5,
) -> torch.Tensor:
  """Which points of paths ``[n, P, 2]`` (``real [n]`` their own) are on a
  corner: the heading turns at least ``min_deg`` from ``half_m`` before the
  point to ``half_m`` after it. ``[n, P]``."""
  k = max(1, round(half_m / resolution))
  i = torch.arange(paths.shape[1], device=paths.device)
  last = (real - 1)[:, None]

  def at(j: torch.Tensor) -> torch.Tensor:
    return paths.gather(1, j.clamp(min=0).minimum(last)[..., None].expand(-1, -1, 2))

  i = i.expand(len(paths), -1)
  before, here, after = at(i - k), at(i), at(i + k)
  a, b = here - before, after - here
  turn = torch.atan2(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0], (a * b).sum(-1))
  inside = (i >= k) & (i <= last - k)
  return inside & (turn.abs() >= math.radians(min_deg))


class PlannerDriver:
  """A stock planner driving ``n`` trials around their phantoms."""

  STATE = ("progress", "stuck_s")
  """The driver's own per-trial state that a cycle changes (``snapshot``)."""

  def __init__(self, n: int, cfg: PlannerDriveCfg, device):
    self.cfg = cfg
    self.n = n
    self.device = device
    self.planner: Nav2Mppi | Dwb = (
      Dwb(cfg.dwb, n, device) if cfg.planner == "dwb" else Nav2Mppi(cfg.nav2, n, device)
    )
    self.boxes = torch.zeros(n, cfg.phantoms.max_count, 5, device=device)
    """Each trial's phantoms (world frame; ``sample_phantoms``)."""
    self.source = torch.zeros(n, dtype=torch.long, device=device)
    """What each trial committed to, an index into ``SOURCES``."""
    self.last: dict[str, Any] = {}
    """The last cycle's ``step`` output."""
    self.v0 = torch.zeros(n, 3, device=device)
    """MPPI: the speed the last cycle's sequences start from."""
    self.lib: dict[str, torch.Tensor] = {}
    points = 1
    if cfg.library is not None:
      z = np.load(cfg.library)
      res = float(z["resolution"])
      assert res == MAP_RESOLUTION == self.planner.cfg.path_resolution, res
      boxes = torch.as_tensor(z["boxes"], device=device)
      self.lib = {
        "boxes": boxes,
        "occupancy": rasterize(boxes, round(float(z["size_m"]) / res), res),
        "paths": torch.as_tensor(z["paths"], device=device),
        "real": torch.as_tensor(z["real"], device=device).long(),
        "map": torch.as_tensor(z["path_map"], device=device).long(),
      }
      self.lib["corners"] = corners(self.lib["paths"], self.lib["real"], res)
      points = self.lib["paths"].shape[1]
      self.boxes = torch.zeros(n, boxes.shape[1], 5, device=device)
    self.path = torch.zeros(n, points, 2, device=device)
    """Each trial's stretch of global path (world)."""
    self.path_real = torch.ones(n, dtype=torch.long, device=device)
    """How many of its points are its own (the rest repeat its end)."""
    self.path_id = torch.zeros(n, dtype=torch.long, device=device)
    """Which library path each trial's stretch is of."""
    self.progress = torch.zeros(n, dtype=torch.long, device=device)
    """The index of the path point each robot is nearest (``local_path``)."""
    self.map_id = torch.zeros(n, dtype=torch.long, device=device)
    self.map_yaw = torch.zeros(n, device=device)
    self.map_anchor = torch.zeros(n, 2, device=device)
    """Where each trial's map is laid: the map point at its spawn, turned by
    ``map_yaw`` (world = R(map_yaw) (map - anchor) + spawn)."""
    self.spawn = torch.zeros(n, 2, device=device)
    self.stuck_s = torch.zeros(n, device=device)
    """How long each walking trial has not moved (``unblock_s``)."""
    self.style = torch.ones(n, STYLE_COLUMNS, device=device)
    """Each trial's style factors (``STYLE_COLUMNS``)."""

  def reset(self, ids: torch.Tensor, start_w: torch.Tensor, goal_w: torch.Tensor):
    """New trials ``ids`` spawned at ``start_w [m, 2]`` facing ``goal_w [m,
    2]``: fresh phantoms between them, or (``library``) a stretch of global
    path laid from the spawn ahead; the planner starts from rest."""
    c, m, dev = self.cfg, len(ids), self.device
    if self.lib:
      self._lay_path(ids, start_w, goal_w)
    else:
      self.boxes[ids] = sample_phantoms(start_w, goal_w, c.phantoms)
    self.planner.reset(ids)
    self.stuck_s[ids] = 0.0

    def uniform(r: tuple[float, float]) -> torch.Tensor:
      return r[0] + (r[1] - r[0]) * torch.rand(m, device=dev)

    lo, hi = c.style_range
    if (lo, hi) != (1.0, 1.0):
      planner = self.planner
      for j, name in enumerate(planner.CRITICS):
        scale = planner.critic_scale.setdefault(name, torch.ones(self.n, device=dev))
        scale[ids] = torch.exp(uniform((math.log(lo), math.log(hi))))
        self.style[ids, j] = scale[ids]
      if isinstance(planner, Nav2Mppi):
        factor = torch.exp(uniform((math.log(lo), math.log(hi))))
        planner.temperature_scale[ids] = factor
        self.style[ids, len(planner.CRITICS)] = factor

  def _lay_path(self, ids: torch.Tensor, start_w: torch.Tensor, goal_w: torch.Tensor):
    """Each trial ``ids`` takes a stretch (``segment_m``) of a random library
    path and its map, turned and moved so the stretch starts at its spawn
    ``start_w``, heading where the robot faces (toward ``goal_w``)."""
    c, lib, m, dev = self.cfg, self.lib, len(ids), self.device
    k = torch.randint(len(lib["real"]), (m,), device=dev)
    real = lib["real"][k]
    lo, hi = (round(x / MAP_RESOLUTION) for x in c.segment_m)
    first = (torch.rand(m, device=dev) * (real - 1 - lo).clamp(min=0)).long()
    span_floor = torch.zeros_like(first)
    if c.corner_prob > 0:
      # Start some stretches shortly before a random corner of their path.
      near, far = (round(x / MAP_RESOLUTION) for x in c.corner_ahead_m)
      # Corners far enough in to start ``near`` before them.
      index = torch.arange(lib["paths"].shape[1], device=dev)
      turn = lib["corners"][k] & (index >= near)
      pick = torch.rand(turn.shape, device=dev) * turn
      corner = pick.argmax(-1)
      back = torch.minimum(torch.randint(near, far + 1, (m,), device=dev), corner)
      biased = (torch.rand(m, device=dev) < c.corner_prob) & turn.any(-1)
      first = torch.where(biased, corner - back, first)
      # ...and walk on at least ``near`` past it.
      past = torch.minimum(corner + near, real - 1)
      span_floor = torch.where(biased, past - first, 0)
    span = torch.maximum(torch.randint(lo, hi + 1, (m,), device=dev), span_floor)
    last = torch.minimum(first + span, real - 1)
    pts = lib["paths"][k]  # [m, P, 2] (map frame)
    idx = torch.minimum(
      first[:, None] + torch.arange(pts.shape[1], device=dev), last[:, None]
    )
    stretch = pts.gather(1, idx[..., None].expand(-1, -1, 2))
    anchor = stretch[:, 0]
    ahead = stretch[:, 4] - anchor
    to_goal = goal_w - start_w
    yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0]) - torch.atan2(
      ahead[:, 1], ahead[:, 0]
    )
    cy, sy = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]

    def to_world(xy: torch.Tensor) -> torch.Tensor:
      rel = xy - anchor[:, None]
      return (
        torch.stack(
          [cy * rel[..., 0] - sy * rel[..., 1], sy * rel[..., 0] + cy * rel[..., 1]], -1
        )
        + start_w[:, None]
      )

    self.path[ids] = to_world(stretch)
    self.path_real[ids] = last - first + 1
    self.path_id[ids] = k
    self.progress[ids] = 0
    mid = lib["map"][k]
    self.map_id[ids], self.map_yaw[ids] = mid, yaw
    self.map_anchor[ids], self.spawn[ids] = anchor, start_w
    boxes = lib["boxes"][mid]
    self.boxes[ids] = torch.cat(
      [to_world(boxes[..., :2]), boxes[..., 2:4], boxes[..., 4:] + yaw[:, None, None]],
      -1,
    )

  def _map_patch(self, pose_w: torch.Tensor) -> torch.Tensor:
    """Each trial's map under the robot at ``pose_w [n, 3]``: its occupancy
    ``[n, map_cells, map_cells]``, robot-centred and yaw-aligned (off the
    map: free)."""
    occ = self.lib["occupancy"]
    cells = occ.shape[-1]
    grid = cell_centres(self.cfg.map_cells, MAP_RESOLUTION, self.device).view(-1, 2)
    c, s = torch.cos(pose_w[:, 2:3]), torch.sin(pose_w[:, 2:3])
    wx = pose_w[:, 0:1] + c * grid[:, 0] - s * grid[:, 1] - self.spawn[:, 0:1]
    wy = pose_w[:, 1:2] + s * grid[:, 0] + c * grid[:, 1] - self.spawn[:, 1:2]
    my, sy = torch.cos(self.map_yaw)[:, None], torch.sin(self.map_yaw)[:, None]
    mx = my * wx + sy * wy + self.map_anchor[:, 0:1]
    mz = -sy * wx + my * wy + self.map_anchor[:, 1:2]
    col = torch.round(mx / MAP_RESOLUTION + (cells - 1) / 2).long()
    row = torch.round(mz / MAP_RESOLUTION + (cells - 1) / 2).long()
    inside = (col >= 0) & (col < cells) & (row >= 0) & (row < cells)
    flat = row.clamp(0, cells - 1) * cells + col.clamp(0, cells - 1)
    hit = occ.view(len(occ), -1)[self.map_id[:, None], flat] & inside
    return hit.view(-1, self.cfg.map_cells, self.cfg.map_cells)

  def cycle(
    self,
    pose_w: torch.Tensor,
    speed_b: torch.Tensor,
    goal_w: torch.Tensor,
    still: torch.Tensor,
    vx_cap: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """One planning cycle for every trial from its pose ``[n, 3]`` (x, y,
    yaw), body speed ``[n, 3]`` (vx, vy, wz) and goal ``[n, 2]``: the
    command ``[n, 3]``. With a ``library`` it tracks each trial's stretch of
    path (``goal_w`` unused: the stretch's end is the goal). ``still`` trials
    stand (0) while their planner
    keeps planning from rest, so it sets off from rest when they do (and
    their own plan, committed at a fault, is that). ``vx_cap [n]`` caps the
    forward command (the pre-fault ``slow`` mode)."""
    c = self.cfg
    if self.lib:
      rows = torch.arange(self.n, device=self.device)
      goal_w = self.path[rows, self.path_real - 1]
      costmap = inflate(
        self._map_patch(pose_w), MAP_RESOLUTION, self.planner.cfg.costmap
      )
    else:
      height = phantom_height(
        self.boxes, pose_w, c.map_cells, MAP_RESOLUTION, c.phantoms.height_m
      )
      costmap = nav2_costmap(height, MAP_RESOLUTION, self.planner.cfg.costmap)
    rel = goal_w - pose_w[:, :2]
    cy, sy = torch.cos(pose_w[:, 2]), torch.sin(pose_w[:, 2])
    goal_b = torch.stack(
      [cy * rel[:, 0] + sy * rel[:, 1], -sy * rel[:, 0] + cy * rel[:, 1]], -1
    )
    speed = torch.where(still[:, None], 0.0, speed_b)
    if c.unblock_s is not None and not self.lib:
      stopped = ~still & (speed[:, :2].norm(dim=-1) < 0.1)
      self.stuck_s = torch.where(stopped, self.stuck_s + self.planner.cfg.model_dt, 0.0)
      free = self.stuck_s >= c.unblock_s
      self.boxes[free] = 0.0
      self.stuck_s[free] = 0.0
    if isinstance(self.planner, Nav2Mppi):
      # The speed MPPI starts this cycle's sequences from (its ``step``).
      p = self.planner.cfg
      acc = torch.tensor([p.ax_max, p.ay_max, p.az_max], device=self.device)
      dec = torch.tensor([p.ax_min, p.ay_min, -p.az_max], device=self.device)
      dt = p.model_dt
      self.v0 = self.planner.last_cmd.clamp(speed + dec * dt, speed + acc * dt)
    path = None
    if self.lib:
      path, self.progress = local_path(
        self.path, self.path_real, self.progress, pose_w, self.planner.cfg
      )
    out = self.planner.step(costmap, MAP_RESOLUTION, goal_b, speed, path=path)
    # A copy: the planner's own tensors change in place later (``reset``).
    self.last = {**out, "plan": out["plan"].clone()}
    # Still trials were sent nothing: their planner starts the next cycle
    # from rest (no warm start winds up), its plan always one from here.
    self.planner.last_cmd[still] = 0.0
    cmd = out["cmd"]
    if vx_cap is not None:
      cmd = torch.cat([torch.minimum(cmd[:, :1], vx_cap[:, None]), cmd[:, 1:]], -1)
    return torch.where(still[:, None], 0.0, cmd)

  def snapshot(self) -> dict[str, torch.Tensor]:
    """The planner's per-trial state and the driver's ``STATE``, for
    ``restore``."""
    out = {
      k: v.clone()
      for k, v in vars(self.planner).items()
      if isinstance(v, torch.Tensor) and v.shape[:1] == (self.n,)
    }
    return out | {k: getattr(self, k).clone() for k in self.STATE}

  def restore(self, saved: dict[str, torch.Tensor], keep: torch.Tensor) -> None:
    """Put back the state ``saved`` for every trial but ``keep``."""
    for k, old in saved.items():
      owner = self if k in self.STATE else self.planner
      now = getattr(owner, k)
      mask = keep.view(-1, *[1] * (old.dim() - 1))
      setattr(owner, k, torch.where(mask, now, old))

  def commit(self, ids: torch.Tensor) -> torch.Tensor:
    """The open-loop plan ``[m, commit_steps, 3]`` trials ``ids`` commit to
    from the last cycle: a random candidate (MPPI: its sequence from the
    command it would send on, the last held; DWB: its twist held) or the
    planner's own plan. Sets ``source``."""
    c, out = self.cfg, self.last
    m, steps = len(ids), c.commit_steps
    cv = out["cv"][ids]  # [m, K, T, 3]
    k = torch.randint(cv.shape[1], (m,), device=self.device)
    pick = cv[torch.arange(m, device=self.device), k]  # [m, T, 3]
    dwb = isinstance(self.planner, Dwb)
    if isinstance(self.planner, Nav2Mppi):
      # MPPI's samples are noise around its plan, unclamped (its constraint
      # critic charges them); committed, a sample is sent as the plan would
      # be: within the speed and acceleration limits.
      pick = self.planner._constrain(pick, self.v0[ids])
      pick = torch.cat([pick[:, 1:], pick[:, -1:]], 1)
    if pick.shape[1] < steps:
      pick = torch.cat([pick, pick[:, -1:].expand(-1, steps - pick.shape[1], -1)], 1)
    own = out["plan"][ids, :steps]
    sample = torch.rand(m, device=self.device) < c.sample_prob
    self.source[ids] = torch.where(sample, 0, 1) + (2 if dwb else 0)
    seq = torch.where(sample[:, None, None], pick[:, :steps], own)
    if c.random_prob > 0:
      rand = torch.rand(m, device=self.device) < c.random_prob
      name = "dwb_random" if dwb else "mppi_random"
      self.source[ids] = torch.where(rand, SOURCES.index(name), self.source[ids])
      seq = torch.where(rand[:, None, None], self._random(ids, steps), seq)
    return seq

  def _random(self, ids: torch.Tensor, steps: int) -> torch.Tensor:
    """Random steps of commands ``[m, steps, 3]`` (``random_prob``)."""
    p, m, dev = self.planner.cfg, len(ids), self.device
    lo, hi = self.cfg.random_hold_s
    ends = torch.randint(lo, hi + 1, (m, steps), device=dev).cumsum(1)
    t = torch.arange(steps, device=dev).expand(m, steps).contiguous()
    hold = torch.searchsorted(ends, t, right=True)  # [m, steps]
    vmin = torch.tensor([p.vx_min, -p.vy_max, -p.wz_max], device=dev)
    vmax = torch.tensor([p.vx_max, p.vy_max, p.wz_max], device=dev)
    values = vmin + (vmax - vmin) * torch.rand(m, steps, 3, device=dev)
    seq = values.gather(1, hold[..., None].expand(-1, -1, 3))
    if isinstance(self.planner, Nav2Mppi):
      seq = self.planner._constrain(seq, self.v0[ids])
      seq = torch.cat([seq[:, 1:], seq[:, -1:]], 1)
    return seq
