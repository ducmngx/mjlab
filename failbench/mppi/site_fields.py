"""The risk critic's distance fields from a prebuilt site map.

Per cycle the runner builds, from the robot's own scan, one capped distance
field per severity level (``cbf.fields_of_levels``): about 17 ms of small
kernels a cycle. A static site doesn't change between cycles, so here the
site is scanned once from above (``scan_site``), its fields are built once
over the whole site (``SiteFields``), and each cycle only crops them at the
robot's pose in one ``grid_sample``.

What differs from the per-cycle fields:

- near the window's edge, distances count obstacles just outside it, which
  the robot's scan can't see (the per-cycle fields read open ground there);
- at headings off the grid the crop is resampled bilinearly, within about a
  cell of the per-cycle field;
- drops are not in them: a ledge is a drop only from the floor the robot
  stands on (``costmap.drop_cells``), so the runner adds the drop field per
  cycle, from its own scan. An obstacle cell that is also a drop (rare: the
  two mark opposite sides of a step) is then priced at both levels instead
  of the drop's.

It needs a static site and the true map (``--sweep-map oracle``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
import torch.nn.functional as F

from failbench.mppi.costmap import FootprintCfg, distance_field, obstacle_cells


@dataclass(frozen=True)
class Site:
  """A top-down scan of the site: ``height [H, W]`` (absolute z, NaN where
  no ray hit) and the geom hit ``geom [H, W]`` (-1: none) at ``(x0, y0) +
  (col, row) * resolution``."""

  height: torch.Tensor
  geom: torch.Tensor
  x0: float
  y0: float
  resolution: float


def scan_site(
  model: mujoco.MjModel,
  bounds: tuple[float, float, float, float],
  resolution: float,
  z0: float,
  geomgroup: tuple[int, ...] = (0,),
) -> Site:
  """Cast a ray straight down from height ``z0`` at every cell of ``bounds``
  ``(xmin, ymin, xmax, ymax)``, seeing only ``geomgroup`` (the runner's
  scanner sees group 0, the terrain), as the scanner does."""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  xmin, ymin, xmax, ymax = bounds
  nx = math.floor((xmax - xmin) / resolution + 1e-6) + 1
  ny = math.floor((ymax - ymin) / resolution + 1e-6) + 1
  group = np.zeros(6, np.uint8)
  group[list(geomgroup)] = 1
  down = np.array([0.0, 0.0, -1.0])
  gid = np.zeros(1, np.int32)
  height = np.full((ny, nx), np.nan, np.float32)
  geom = np.full((ny, nx), -1, np.int64)
  p = np.array([0.0, 0.0, z0])
  for j in range(ny):
    p[1] = ymin + j * resolution
    for i in range(nx):
      p[0] = xmin + i * resolution
      d = mujoco.mj_ray(model, data, p, down, group, True, -1, gid)
      if d >= 0:
        height[j, i] = z0 - d
        geom[j, i] = gid[0]
  return Site(torch.from_numpy(height), torch.from_numpy(geom), xmin, ymin, resolution)


def site_bounds(
  model: mujoco.MjModel, margin: float, geomgroup: int = 0
) -> tuple[float, float, float, float]:
  """``(xmin, ymin, xmax, ymax)`` round every ``geomgroup`` geom but planes,
  ``margin`` (m) out."""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  keep = (model.geom_group == geomgroup) & (
    model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE
  )
  if not keep.any():
    raise ValueError("No geoms to bound the site by.")
  xy, r = data.geom_xpos[keep, :2], model.geom_rbound[keep, None]
  lo, hi = (xy - r).min(0) - margin, (xy + r).max(0) + margin
  return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


class SiteFields:
  """The site's distance field per severity level, built once, for ``n``
  robots with their own severities per geom ``severity [n, ngeom + 1]``
  (the last entry for misses)."""

  def __init__(
    self,
    site: Site,
    severity: torch.Tensor,
    resolution: float,
    fp: FootprintCfg,
  ):
    if not math.isclose(site.resolution, resolution):
      raise ValueError("The site scan and the planner's map differ in resolution.")
    dev = severity.device
    height = site.height.to(dev)
    obst = obstacle_cells(height[None], fp.wall_jump_m)[0]
    sev = severity[:, site.geom.to(dev) % severity.shape[1]]  # [n, H, W]
    level = torch.where(obst, sev, 0.0)
    self.values = [v for v in level.unique().tolist() if v > 0]
    n, h, w = level.shape
    masks = torch.stack([level == v for v in self.values], 1) if self.values else None
    self.max_dist = fp.max_dist_m
    self.fields = (
      distance_field(masks.flatten(0, 1), resolution, fp.max_dist_m).view(
        n, len(self.values), h, w
      )
      if masks is not None
      else None
    )
    self.site = site
    self.resolution = resolution

  def sample(
    self, xy: torch.Tensor, yaw: torch.Tensor, cells: int
  ) -> list[tuple[float, torch.Tensor]]:
    """(severity, field ``[n, cells, cells]``) per level within reach of
    each robot's yaw-aligned window centred at ``xy [n, 2]`` (rows along
    its y), as ``cbf.fields_of_levels`` of the robot's own scan."""
    if self.fields is None:
      return []
    s, res = self.site, self.resolution
    dev = self.fields.device
    xy, yaw = xy.to(dev, torch.float32), yaw.to(dev, torch.float32)
    off = (torch.arange(cells, device=dev, dtype=torch.float32) - (cells - 1) / 2) * res
    gy, gx = torch.meshgrid(off, off, indexing="ij")  # [cells, cells]
    c, sn = yaw.cos()[:, None, None], yaw.sin()[:, None, None]
    wx = xy[:, 0, None, None] + c * gx - sn * gy
    wy = xy[:, 1, None, None] + sn * gx + c * gy
    h, w = self.fields.shape[-2:]
    grid = torch.stack(
      [(wx - s.x0) / (res * (w - 1)) * 2 - 1, (wy - s.y0) / (res * (h - 1)) * 2 - 1],
      -1,
    )
    # Off the site: nothing there (zeros padding of field - max).
    out = (
      F.grid_sample(
        self.fields - self.max_dist, grid, align_corners=True, padding_mode="zeros"
      )
      + self.max_dist
    )
    near = (out < self.max_dist - 1e-4).flatten(2).any(-1).any(0).tolist()
    return [(v, out[:, k]) for k, v in enumerate(self.values) if near[k]]
