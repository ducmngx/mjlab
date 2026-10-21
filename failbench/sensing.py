"""What a Go2's head LiDAR could have seen of a clean elevation map.

The simulator's map is cast straight down from above the trunk, so it sees
behind step edges and under the robot. ``visible_cells`` marks which cells
a head-mounted LiDAR (Unitree L1) would actually return, from one or more
sensor poses (the union of several recent poses is a fused map, like
elevation_mapping_cupy accumulating scans while the robot walks):

- terrain occlusion: a cell is hidden if the terrain rises above the line
  of sight from the sensor to it (ray-marched on the heightmap itself);
- field of view: a cone of ``fov_half_angle`` around the sensor axis, and
  a minimum range;
- self-occlusion: cells under the trunk's footprint at that pose.

Sensor numbers are ``LidarCfg`` defaults. The mount pose is from Unitree's
``go2_description`` URDF (``radar_joint``: xyz (0.28945, 0, -0.046825),
rpy (0, 2.8782, 0)); the L1's 360 x 90 deg field of view is modelled as a
hemisphere about its axis. Both should be checked against Unitree's
datasheet before they're quoted.

``corrupt_map`` adds range noise, a per-map height offset and dropout, for
use at training time so every epoch sees fresh noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import quat_apply


@dataclass(frozen=True)
class LidarCfg:
  mount_pos_b: tuple[float, float, float] = (0.28945, 0.0, -0.046825)
  mount_pitch: float = 2.8782
  """Rotation about the base y axis (rad) taking the sensor's +z (its
  axis) into the base frame."""
  fov_half_angle: float = math.pi / 2
  min_range: float = 0.05
  occlusion_tol: float = 0.03
  """Terrain must rise this far (m) above the line of sight to hide a cell."""
  march_samples: int = 48
  body_footprint: tuple[float, float, float, float] = (-0.35, 0.25, -0.15, 0.15)
  """(x_lo, x_hi, y_lo, y_hi) in the base frame: ground hidden by the trunk."""


@dataclass(frozen=True)
class NoiseCfg:
  height_std: float = 0.02
  offset_std: float = 0.03
  dropout: float = 0.05


def sensor_poses(
  base_pos: torch.Tensor, base_quat: torch.Tensor, lidar: LidarCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sensor position and unit axis ``[..., 3]`` from base poses ``[..., 3/4]``
  (any common frame)."""
  mount = torch.tensor(lidar.mount_pos_b, device=base_pos.device)
  c, s = math.cos(lidar.mount_pitch), math.sin(lidar.mount_pitch)
  axis_b = torch.tensor([s, 0.0, c], device=base_pos.device)
  pos = base_pos + quat_apply(base_quat, mount.expand_as(base_pos))
  axis = quat_apply(base_quat, axis_b.expand_as(base_pos))
  return pos, axis


def cell_centres(cells: int, resolution: float, device) -> torch.Tensor:
  """``[cells, cells, 2]`` (x, y) of each cell, row = y, like ``GridSpec``."""
  r = (torch.arange(cells, device=device) - (cells - 1) / 2) * resolution
  y, x = torch.meshgrid(r, r, indexing="ij")
  return torch.stack([x, y], dim=-1)


def visible_cells(
  height: torch.Tensor,
  sensor_pos: torch.Tensor,
  sensor_axis: torch.Tensor,
  base_pos: torch.Tensor,
  base_yaw: torch.Tensor,
  resolution: float,
  lidar: LidarCfg | None = None,
) -> torch.Tensor:
  """Cells of ``height [B, H, W]`` seen from any of ``P`` poses.

  ``sensor_pos`` / ``sensor_axis`` / ``base_pos`` are ``[B, P, 3]`` and
  ``base_yaw`` ``[B, P]``, all in the map's frame (heights are z in that
  frame; NaN = no terrain, which neither shows nor occludes). Returns
  ``[B, H, W]`` bool."""
  lidar = lidar or LidarCfg()
  b, h, w = height.shape
  dev = height.device
  xy = cell_centres(h, resolution, dev).view(1, 1, h * w, 2)
  z = height.view(b, 1, h * w)
  target = torch.cat([xy.expand(b, 1, -1, -1), z[..., None]], dim=-1)  # [B,1,C,3]
  src = sensor_pos[:, :, None, :]  # [B,P,1,3]
  ray = target - src  # [B,P,C,3]
  dist = ray.norm(dim=-1)
  direction = ray / dist.clamp(min=1e-6)[..., None]
  cos = (direction * sensor_axis[:, :, None, :]).sum(-1)
  seen = (cos >= math.cos(lidar.fov_half_angle)) & (dist >= lidar.min_range)

  # Trunk footprint at each pose hides the ground under it.
  rel = xy - base_pos[:, :, None, :2]
  cy, sy = torch.cos(base_yaw)[..., None], torch.sin(base_yaw)[..., None]
  bx = cy * rel[..., 0] + sy * rel[..., 1]
  by = -sy * rel[..., 0] + cy * rel[..., 1]
  x_lo, x_hi, y_lo, y_hi = lidar.body_footprint
  seen &= ~((bx > x_lo) & (bx < x_hi) & (by > y_lo) & (by < y_hi))

  # Ray march, skipping the last cell so a cell never hides itself.
  k = lidar.march_samples
  u = (torch.arange(k, device=dev) + 0.5) / k  # [K]
  stop = 1.0 - resolution / ray[..., :2].norm(dim=-1).clamp(min=resolution)
  uu = u.view(1, 1, 1, k) * stop[..., None]  # [B,P,C,K]
  pts = src[..., None, :] + uu[..., None] * ray[..., None, :]  # [B,P,C,K,3]
  ix = torch.round(pts[..., 0] / resolution + (w - 1) / 2).long()
  iy = torch.round(pts[..., 1] / resolution + (h - 1) / 2).long()
  inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
  flat = (iy.clamp(0, h - 1) * w + ix.clamp(0, w - 1)).view(b, -1)
  ground = torch.gather(height.view(b, h * w), 1, flat).view_as(ix)
  blocked = inside & (ground > pts[..., 2] + lidar.occlusion_tol)
  seen &= ~blocked.any(dim=-1)
  seen &= ~torch.isnan(z)
  return seen.any(dim=1).view(b, h, w)


def corrupt_map(
  height: torch.Tensor,
  known: torch.Tensor,
  noise: NoiseCfg | None = None,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """Noisy 2-channel map ``[..., 2, H, W]``: height (0 where unknown) and the
  known mask, from a clean ``height`` and visibility ``known [..., H, W]``."""
  noise = noise or NoiseCfg()
  dev, shape = height.device, height.shape
  offset_shape = (*shape[:-2], 1, 1)
  offset = torch.randn(offset_shape, generator=generator, device=dev)
  noisy = height + offset * noise.offset_std
  noisy = noisy + torch.randn(shape, generator=generator, device=dev) * noise.height_std
  drop = torch.rand(shape, generator=generator, device=dev) < noise.dropout
  keep = known & ~drop & ~torch.isnan(height)
  noisy = torch.where(keep, noisy, 0.0)
  return torch.stack([noisy, keep.to(height.dtype)], dim=-3)
