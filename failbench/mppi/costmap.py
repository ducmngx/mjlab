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
- ``drop_cells``: a cell is a drop (a ledge's foot, where a robot that goes
  over falls) if, within ``drop_reach_m``, ground at the robot's own level
  (within ``drop_level_m`` of the floor under the map's centre) stands more
  than ``drop_m`` above it: more than two stair risers, so stairs down
  aren't drops. Taking only ground at the robot's level keeps wall and
  crate tops out, which the LiDAR often sees with the floor beside them
  hidden. The rim itself is already an obstacle (it stands above the foot).
  The foot of a ledge is often hidden from the LiDAR too, so this needs the
  floor below to have been seen.
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

- ``hit_prob``: the chance a footprint touches an obstacle at each step,
  for a predicted pose with Gaussian spread: per circle,
  ``Phi((radius - clearance) / spread)`` with the same spread as ``sweep``'s
  inflation (over ``k_sigma``), the largest over circles. Far from anything
  it is 0, so open ground carries no risk.

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
  drop_m: float = 0.45
  drop_reach_m: float = 0.3
  drop_level_m: float = 0.2
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


def _pool(h: torch.Tensor, r: int, op: str) -> torch.Tensor:
  """Max (``op="max"``) or min over ``(2r+1)^2`` windows of ``[B, H, W]``;
  unknown cells (NaN) are skipped, and an all-unknown window is NaN."""
  sign = 1.0 if op == "max" else -1.0
  x = (sign * h).nan_to_num(-torch.inf)[:, None]
  out = F.max_pool2d(x, 2 * r + 1, 1, r)[:, 0]
  return torch.where(out.isinf(), float("nan"), sign * out)


def drop_cells(height: torch.Tensor, cfg: "FootprintCfg", resolution: float):
  """``height [B, H, W]`` (NaN unknown), centred on the robot -> drop mask
  ``[B, H, W]``."""
  med = median3(height)
  h, w = med.shape[-2:]
  r = round(cfg.drop_reach_m / resolution)
  floor = med[:, h // 2 - r : h // 2 + r + 1, w // 2 - r : w // 2 + r + 1]
  floor = floor.flatten(1).nanmedian(-1).values[:, None, None]
  level = torch.where((med - floor).abs() < cfg.drop_level_m, med, float("nan"))
  return (_pool(level, r, "max") - med) > cfg.drop_m  # NaN compares False.


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


def hit_prob(
  dist: torch.Tensor,
  pose: torch.Tensor,
  std: torch.Tensor,
  resolution: float,
  cfg: FootprintCfg,
  radius: float | None = None,
  min_std: float = 0.02,
) -> torch.Tensor:
  """P(the footprint along ``pose [B, K, T, 3]``, spread ``std``, touches an
  obstacle of ``dist [B, H, W]``) per step ``[B, K, T]``; ``radius``
  overrides the circles' (e.g. a robot lying down)."""
  ox = torch.tensor(cfg.circles_x, device=pose.device)
  yaw = pose[..., 2:3]
  centres = pose[..., None, :2] + ox[:, None] * torch.stack(
    [torch.cos(yaw), torch.sin(yaw)], dim=-1
  )
  clear = cfg.max_dist_m - _sample(cfg.max_dist_m - dist, centres, resolution)
  spread = std[..., :2].norm(dim=-1, keepdim=True) + ox.abs() * std[..., 2:3]
  r = cfg.radius if radius is None else radius
  z = (r - clear) / spread.clamp(min=min_std)
  return torch.special.ndtr(z).amax(-1)
