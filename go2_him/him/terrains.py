"""HIMLoco's training terrain mix, built from mjlab terrain presets.

HIMLoco (legged_gym ``terrain.py`` with ``terrain_proportions`` [0.1, 0.2,
0.3, 0.3, 0.1]) uses a 10 x 20 grid of 8 m tiles in curriculum layout:

========== ====== =============================================================
type       share  per difficulty d in [0, 1]
========== ====== =============================================================
smooth     0.1    pyramid slope, d * 0.4 rise per run, half of them going down
slope
rough      0.2    the same slope plus uniform noise of +-5 cm (0.5 cm steps)
slope
stairs up  0.3    pyramid stairs, riser 0.05 + 0.18 d m, 0.31 m tread
stairs     0.3    the same, descending
down
discrete   0.1    20 random 1-2 m blocks, 0.05 + 0.2 d m high
========== ====== =============================================================

``HfRoughSlopeTerrainCfg`` is the one tile mjlab has no preset for.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.config import (
  discrete_obstacles,
  hf_pyramid_slope,
  hf_pyramid_slope_inv,
  pyramid_stairs,
  pyramid_stairs_inv,
)
from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

_SLOPE_RANGE = (0.0, 0.4)
_STEP_RANGE = (0.05, 0.23)


@dataclass(kw_only=True)
class HfRoughSlopeTerrainCfg(SubTerrainCfg):
  """A pyramid slope with uniform height noise on top (legged_gym's
  ``pyramid_sloped_terrain`` followed by ``random_uniform_terrain``)."""

  slope_range: tuple[float, float] = _SLOPE_RANGE
  inverted: bool = False
  noise_range: tuple[float, float] = (-0.05, 0.05)
  noise_step: float = 0.005
  noise_downsample: float = 0.2
  """Noise is drawn on a grid this coarse (m) and interpolated."""
  platform_width: float = 3.0
  horizontal_scale: float = 0.1

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    res = self.horizontal_scale
    nx, ny = int(self.size[0] / res), int(self.size[1] / res)
    slope = self.slope_range[0] + difficulty * (
      self.slope_range[1] - self.slope_range[0]
    )
    if self.inverted:
      slope = -slope

    # Pyramid: height rises linearly to the centre, flat-topped platform.
    x = np.arange(nx)
    y = np.arange(ny)
    cx, cy = nx / 2, ny / 2
    fx = (cx - np.abs(cx - x)) / cx
    fy = (cy - np.abs(cy - y)) / cy
    height_max = slope * self.size[0] / 2
    heights = height_max * fx[:, None] * fy[None, :]
    pf = int(self.platform_width / res / 2)
    z_pf = heights[int(cx) - pf, int(cy) - pf]
    heights = np.clip(heights, min(0.0, z_pf), max(0.0, z_pf))

    # Coarse uniform noise, quantized and upsampled (nearest) to the grid.
    step = max(1, int(round(self.noise_downsample / res)))
    levels = np.arange(self.noise_range[0], self.noise_range[1] + 1e-9, self.noise_step)
    coarse = rng.choice(levels, size=(nx // step + 1, ny // step + 1))
    heights = heights + np.kron(coarse, np.ones((step, step)))[:nx, :ny]

    z_min, z_max = float(heights.min()), float(heights.max())
    span = max(z_max - z_min, 1e-3)
    normalized = ((heights - z_min) / span).astype(np.float32)
    uid = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"hfield_{uid}",
      size=[self.size[0] / 2, self.size[1] / 2, span, span],
      nrow=nx,
      ncol=ny,
      userdata=normalized.flatten().tolist(),
    )
    material = color_by_height(spec, normalized, uid, heights)
    geom = spec.body("terrain").add_geom(
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[self.size[0] / 2, self.size[1] / 2, z_min],
      material=material,
    )
    origin = np.array(
      [self.size[0] / 2, self.size[1] / 2, float(heights[nx // 2, ny // 2])]
    )
    return TerrainOutput(
      origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=field)]
    )


HIM_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=25.0,
  num_rows=10,
  num_cols=20,
  curriculum=True,
  sub_terrains={
    "slope_up": hf_pyramid_slope(
      proportion=0.05, slope_range=_SLOPE_RANGE, platform_width=3.0
    ),
    "slope_down": hf_pyramid_slope_inv(
      proportion=0.05, slope_range=_SLOPE_RANGE, platform_width=3.0
    ),
    "rough_slope_up": HfRoughSlopeTerrainCfg(proportion=0.1),
    "rough_slope_down": HfRoughSlopeTerrainCfg(proportion=0.1, inverted=True),
    "stairs_up": pyramid_stairs_inv(
      proportion=0.3, step_height_range=_STEP_RANGE, step_width=0.31
    ),
    "stairs_down": pyramid_stairs(
      proportion=0.3, step_height_range=_STEP_RANGE, step_width=0.31
    ),
    "discrete": discrete_obstacles(
      proportion=0.1,
      obstacle_width_range=(1.0, 2.0),
      obstacle_height_range=(0.05, 0.25),
      num_obstacles=20,
      platform_width=3.0,
    ),
  },
  add_lights=True,
)
"""HIMLoco's terrain mix. In curriculum mode each type is one column and
``proportion`` sets how many robots start on it; the training curriculum
moves robots between difficulty rows."""
