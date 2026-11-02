"""Phantom obstacles: boxes only the planner sees.

In the planner-driven data mode (``failbench/driver.py``) a stock planner
drives each trial toward its goal. On the training terrains there is nothing
to walk around, so it would walk straight; each trial instead draws a few
boxes near the spawn-to-goal line that go into the planner's costmap and
nowhere else. The planner sidesteps, turns, slows and stops for them as it
does in real scenes, while the physics (and so the drift the impact model
learns) stays obstacle-free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from failbench.sensing import cell_centres


@dataclass(frozen=True)
class PhantomCfg:
  count: tuple[int, int] = (1, 6)
  """Boxes in a trial with phantoms (uniform, inclusive)."""
  none_prob: float = 0.2
  """Share of trials with no phantoms (the planner walks straight)."""
  size_m: tuple[float, float] = (0.2, 1.2)
  """Each side's length (uniform)."""
  lateral_m: float = 1.5
  """A box centre is at most this far from the spawn-to-goal line."""
  along_m: tuple[float, float] = (0.8, 4.5)
  """...and this far along it from the spawn: where a robot walks in the 2-4
  s before its fault, so the planner is going round them when it fires."""
  clear_m: float = 0.6
  """No box comes this close to the spawn or the goal."""
  height_m: float = 0.8
  max_count: int = 6
  tries: int = 20
  """Redraws of a box too close to the spawn or goal before it is dropped."""


def _box_distance(boxes: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
  """Distance from points ``xy [n, 2]`` to each box ``[n, M, 5]`` (cx, cy,
  half x, half y, yaw), 0 inside: ``[n, M]``."""
  rel = xy[:, None] - boxes[..., :2]
  c, s = torch.cos(boxes[..., 4]), torch.sin(boxes[..., 4])
  local = torch.stack(
    [c * rel[..., 0] + s * rel[..., 1], -s * rel[..., 0] + c * rel[..., 1]], -1
  )
  return (local.abs() - boxes[..., 2:4]).clamp(min=0).norm(dim=-1)


def sample_phantoms(
  start_w: torch.Tensor, goal_w: torch.Tensor, cfg: PhantomCfg
) -> torch.Tensor:
  """Each trial's boxes ``[n, max_count, 5]`` (centre x, y, half x, half y,
  yaw; world frame) near the start of its spawn-to-goal line (``along_m``),
  clear of both ends.
  Unused slots have half sizes 0."""
  n, m, dev = start_w.shape[0], cfg.max_count, start_w.device
  d = goal_w - start_w
  length = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
  unit = d / length
  normal = torch.stack([-unit[:, 1], unit[:, 0]], -1)

  def draw() -> torch.Tensor:
    lo_m, hi_m = cfg.along_m
    hi = torch.minimum(length, torch.full_like(length, hi_m))
    lo = torch.minimum(hi, torch.full_like(hi, lo_m))
    along = lo + torch.rand(n, m, device=dev) * (hi - lo)
    side = (2 * torch.rand(n, m, device=dev) - 1) * cfg.lateral_m
    centre = start_w[:, None] + along[..., None] * unit[:, None]
    centre = centre + side[..., None] * normal[:, None]
    lo, hi = cfg.size_m
    half = (lo + (hi - lo) * torch.rand(n, m, 2, device=dev)) / 2
    yaw = (2 * torch.rand(n, m, 1, device=dev) - 1) * math.pi
    return torch.cat([centre, half, yaw], -1)

  boxes = draw()
  ok = torch.zeros(n, m, dtype=torch.bool, device=dev)
  for _ in range(cfg.tries):
    clear = (_box_distance(boxes, start_w) > cfg.clear_m) & (
      _box_distance(boxes, goal_w) > cfg.clear_m
    )
    ok = ok | clear
    if ok.all():
      break
    boxes = torch.where(ok[..., None], boxes, draw())
  lo, hi = cfg.count
  count = torch.randint(lo, hi + 1, (n, 1), device=dev)
  count = torch.where(torch.rand(n, 1, device=dev) < cfg.none_prob, 0, count)
  used = ok & (torch.arange(m, device=dev) < count)
  unused = boxes * torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0], device=dev)
  return torch.where(used[..., None], boxes, unused)


def phantom_height(
  boxes: torch.Tensor,
  pose_w: torch.Tensor,
  cells: int,
  resolution: float,
  height_m: float,
) -> torch.Tensor:
  """Robot-centred, yaw-aligned height maps ``[n, cells, cells]`` of the
  boxes ``[n, M, 5]`` around robot poses ``pose_w [n, 3]`` (x, y, yaw):
  ``height_m`` inside a box, 0 elsewhere."""
  grid = cell_centres(cells, resolution, pose_w.device).view(-1, 2)  # [C, 2]
  c, s = torch.cos(pose_w[:, 2:3]), torch.sin(pose_w[:, 2:3])
  wx = pose_w[:, 0:1] + c * grid[:, 0] - s * grid[:, 1]
  wy = pose_w[:, 1:2] + s * grid[:, 0] + c * grid[:, 1]
  rel_x = wx[:, None] - boxes[..., 0:1]  # [n, M, C]
  rel_y = wy[:, None] - boxes[..., 1:2]
  bc, bs = torch.cos(boxes[..., 4:5]), torch.sin(boxes[..., 4:5])
  lx = bc * rel_x + bs * rel_y
  ly = -bs * rel_x + bc * rel_y
  inside = (lx.abs() <= boxes[..., 2:3]) & (ly.abs() <= boxes[..., 3:4])
  inside = inside & (boxes[..., 2:3] > 0)
  h = inside.any(1).float() * height_m
  return h.view(-1, cells, cells)
