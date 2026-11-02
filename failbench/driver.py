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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from failbench.mppi.dwb import Dwb, DwbCfg
from failbench.mppi.nav2 import Nav2Mppi, Nav2MppiCfg, nav2_costmap
from failbench.phantoms import PhantomCfg, phantom_height, sample_phantoms

MAP_RESOLUTION = 0.05

SOURCES = ("mppi_sample", "mppi_plan", "dwb_twist", "dwb_best")
"""What a trial committed to at its fault (``PlannerDriver.source``)."""


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
  commit_steps: int = 30
  """Planning steps (0.1 s) a commit runs open loop: 3 s."""


class PlannerDriver:
  """A stock planner driving ``n`` trials around their phantoms."""

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

  def reset(self, ids: torch.Tensor, start_w: torch.Tensor, goal_w: torch.Tensor):
    """New trials ``ids``: fresh phantoms between ``start_w [m, 2]`` and
    ``goal_w [m, 2]``; the planner starts from rest."""
    self.boxes[ids] = sample_phantoms(start_w, goal_w, self.cfg.phantoms)
    self.planner.reset(ids)

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
    command ``[n, 3]``. ``still`` trials stand (0) while their planner
    keeps planning from rest, so it sets off from rest when they do (and
    their own plan, committed at a fault, is that). ``vx_cap [n]`` caps the
    forward command (the pre-fault ``slow`` mode)."""
    c = self.cfg
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
    if isinstance(self.planner, Nav2Mppi):
      # The speed MPPI starts this cycle's sequences from (its ``step``).
      p = self.planner.cfg
      acc = torch.tensor([p.ax_max, p.ay_max, p.az_max], device=self.device)
      dec = torch.tensor([p.ax_min, p.ay_min, -p.az_max], device=self.device)
      dt = p.model_dt
      self.v0 = self.planner.last_cmd.clamp(speed + dec * dt, speed + acc * dt)
    out = self.planner.step(costmap, MAP_RESOLUTION, goal_b, speed)
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
    """The planner's per-trial state, for ``restore``."""
    return {
      k: v.clone()
      for k, v in vars(self.planner).items()
      if isinstance(v, torch.Tensor) and v.shape[:1] == (self.n,)
    }

  def restore(self, saved: dict[str, torch.Tensor], keep: torch.Tensor) -> None:
    """Put back the planner state ``saved`` for every trial but ``keep``."""
    for k, old in saved.items():
      now = getattr(self.planner, k)
      mask = keep.view(-1, *[1] * (old.dim() - 1))
      setattr(self.planner, k, torch.where(mask, now, old))

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
    return torch.where(sample[:, None, None], pick[:, :steps], own)
