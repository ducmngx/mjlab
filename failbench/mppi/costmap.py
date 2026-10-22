"""Geometric obstacle check: the robot's footprint swept along a predicted
path over the elevation map.

- ``obstacle_cells``: a cell is an obstacle if it stands more than
  ``wall_jump_m`` above a neighbouring cell, i.e. a wall, crate or pole
  edge rather than a stair or a slope (course risers are at most 0.14 m,
  crates at least 0.2 m). Absolute height can't be used: on stairs the
  ground ahead is legitimately a metre up. Map noise would turn stair
  edges into walls, so jumps are taken on a 3x3 median of the map, which
  keeps the edges of anything 2 cells or wider; thin poles, which the
  median erases, are caught as cells standing ``wall_jump_m`` above their
  neighbourhood's median. With the training map noise (2 cm per cell) this
  flags a 0.14 m staircase in about 4% of maps and a 0.2 m crate in 99%.
  Unknown cells (NaN) are never obstacles.
- ``distance_field``: distance (m) from each cell to the nearest obstacle,
  capped at ``max_dist_m``, by chamfer propagation over 8-neighbours
  (within about 8% of Euclidean; cheap on a GPU and on a Jetson).
- ``sweep``: the trunk as circles along its axis. Each circle's radius
  grows by ``k_sigma * sqrt(sx^2 + sy^2)``, which covers the predicted
  ``+-k sigma`` box exactly, plus ``|offset| * k_sigma * s_yaw`` for heading
  spread (``failbench/README.md``, "Drift uncertainty"), capped at
  ``max_dist_m`` (beyond it, open ground can't be told from a far wall).
  Returns how far
  each circle reaches into an obstacle and whether its centre is on an
  unknown cell. Off the map counts as free and unknown.

Maps are robot-centred and yaw-aligned with the plan's start frame, on the
grid of ``failbench/sensing.cell_centres`` (row = y).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class FootprintCfg:
  circles_x: tuple[float, ...] = (-0.22, 0.0, 0.22)
  """Circle centres along the trunk axis (m, base frame)."""
  radius: float = 0.2
  """Covers the Go2's legs (about 0.4 m wide) and a 0.84 m length."""
  k_sigma: float = 2.0
  wall_jump_m: float = 0.18
  max_dist_m: float = 1.5
  """Distances are known up to this; an inflated circle is capped to it, so
  a very uncertain plan in open ground isn't charged for walls that aren't
  there."""


def median3(height: torch.Tensor) -> torch.Tensor:
  """3x3 median of ``[B, H, W]``, ignoring NaN (NaN where all are)."""
  pad = F.pad(height[:, None], (1, 1, 1, 1), value=float("nan"))
  return F.unfold(pad, 3).nanmedian(dim=1).values.view_as(height)


def obstacle_cells(height: torch.Tensor, wall_jump_m: float) -> torch.Tensor:
  """``height [B, H, W]`` (NaN unknown) -> obstacle mask ``[B, H, W]``."""
  med = median3(height)
  return _jumps(med, wall_jump_m) | ((height - med) > wall_jump_m)


def _jumps(h: torch.Tensor, wall_jump_m: float) -> torch.Tensor:
  pad = F.pad(h[:, None], (1, 1, 1, 1), value=float("nan"))[:, 0]
  obst = torch.zeros_like(h, dtype=torch.bool)
  for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    nb = pad[:, 1 + dy : pad.shape[1] - 1 + dy, 1 + dx : pad.shape[2] - 1 + dx]
    obst |= (h - nb) > wall_jump_m  # NaN compares False.
  return obst


def distance_field(
  obst: torch.Tensor, resolution: float, max_dist: float
) -> torch.Tensor:
  """``[B, H, W]`` distance (m) to the nearest obstacle cell, capped."""
  d = torch.where(obst, 0.0, max_dist)
  h, w = d.shape[-2:]
  diag = resolution * math.sqrt(2.0)
  for _ in range(math.ceil(max_dist / resolution)):
    p = F.pad(d[:, None], (1, 1, 1, 1), value=max_dist)[:, 0]
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        if dy == 0 and dx == 0:
          continue
        step = diag if dy and dx else resolution
        d = torch.minimum(d, p[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w] + step)
  return d.clamp(max=max_dist)


def _sample(grid: torch.Tensor, xy: torch.Tensor, resolution: float) -> torch.Tensor:
  """Bilinear ``grid [B, H, W]`` at metric points ``xy [B, ..., 2]``; zero
  off the map."""
  b, h, w = grid.shape
  pts = xy.reshape(b, -1, 1, 2)
  gx = pts[..., 0] / resolution / ((w - 1) / 2)
  gy = pts[..., 1] / resolution / ((h - 1) / 2)
  out = F.grid_sample(
    grid[:, None], torch.stack([gx, gy], dim=-1), align_corners=True
  )  # [B, 1, P, 1]
  return out.view(*xy.shape[:-1])


def sweep(
  dist: torch.Tensor,
  known: torch.Tensor,
  pose: torch.Tensor,
  std: torch.Tensor,
  resolution: float,
  cfg: FootprintCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Penetration (m) and unknown flag, ``[B, K, T, C]``, of footprint
  circles along ``pose [B, K, T, 3]`` with spread ``std [B, K, T, 3]``
  over ``dist`` / ``known [B, H, W]``."""
  ox = torch.tensor(cfg.circles_x, device=pose.device)
  yaw = pose[..., 2:3]
  centres = pose[..., None, :2] + ox[:, None] * torch.stack(
    [torch.cos(yaw), torch.sin(yaw)], dim=-1
  )  # [B, K, T, C, 2]
  # Off the map: distance max_dist (sampled as max_dist - d, zero padded).
  clear = cfg.max_dist_m - _sample(cfg.max_dist_m - dist, centres, resolution)
  unknown = _sample(known.float(), centres, resolution) < 0.5
  grow = cfg.k_sigma * (
    std[..., :2].norm(dim=-1, keepdim=True) + ox.abs() * std[..., 2:3]
  )
  reach = (cfg.radius + grow).clamp(max=cfg.max_dist_m)
  return F.relu(reach - clear), unknown
