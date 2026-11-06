"""The household layouts (``failbench/house.py``) as MuJoCo terrain, and
their routes for the runner.

One tile per layout, the flat's lower-left corner at the tile origin plus
``margin``. Every item is a box named ``course_<class>_<n>`` so
``failbench/severity.py`` prices it (``floor`` items cost nothing). Flat
floor items (slabs, platforms, stair treads) are boxes, so step edges stay
sharp; ground with ramps or rough patches is one heightfield sampled from
``ground_height``, the same surface the planners' maps are drawn from,
with the base slab cut away under it so dips below 0 are real. Objects
sink ``EMBED_M`` into the ground so none floats over a dip.

``route_table`` gives each route's start (x, y, heading along its global
path, ground height), its goal, and its global path, in the layout frame;
the runner assigns robot ``i`` route ``i % routes``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from failbench.house import LAYOUTS, Item, ground_height, plan_routes
from failbench.l_course import FLOOR_RGBA
from failbench.severity_course import CLASS_RGBA
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

EMBED_M = 0.1
DECAL_M = 0.002
"""Flat floor boxes over the heightfield stand this far above it."""
WALL_RGBA = (0.55, 0.54, 0.52, 1.0)
PLATFORM_RGBA = (0.62, 0.60, 0.55, 1.0)
HFIELD_BASE_M = 0.2
HEADING_AHEAD_M = 0.4
"""The spawn faces the global path's point this far along it."""

_names = itertools.count()


def _shaped(it: Item) -> bool:
  return it.cls == "floor" and (bool(it.ramp) or it.rough > 0)


def _overlaps(it: Item, rect) -> bool:
  x0, x1, y0, y1 = rect
  return it.x0 < x1 and x0 < it.x1 and it.y0 < y1 and y0 < it.y1


def _rects_minus(
  outer: tuple[float, float, float, float], hole: tuple[float, float, float, float]
) -> list[tuple[float, float, float, float]]:
  """``outer`` (x0, x1, y0, y1) less ``hole``: up to four strips."""
  x0, x1, y0, y1 = outer
  hx0, hx1, hy0, hy1 = hole
  strips = [
    (x0, hx0, y0, y1),
    (hx1, x1, y0, y1),
    (hx0, hx1, y0, hy0),
    (hx0, hx1, hy1, y1),
  ]
  return [r for r in strips if r[1] - r[0] > 1e-6 and r[3] - r[2] > 1e-6]


@dataclass(kw_only=True)
class HouseTerrainCfg(SubTerrainCfg):
  layout: str = "apartment"
  margin: float = 1.0
  """Floor round the flat, inside the tile."""
  hfield_resolution: float = 0.05

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng
    lay = LAYOUTS[self.layout]
    body = spec.body("terrain")
    m = self.margin
    out: list[TerrainGeometry] = []

    def box(cls, x0, x1, y0, y1, z0, z1, rgba):
      geom = body.add_geom(
        name=f"course_{cls}_{next(_names)}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
        pos=(m + (x0 + x1) / 2, m + (y0 + y1) / 2, (z0 + z1) / 2),
      )
      geom.rgba[:] = rgba
      out.append(TerrainGeometry(geom=geom, color=rgba))

    w, h = lay.size
    tile = (-m, w + m, -m, h + m)
    shaped = [it for it in lay.items if _shaped(it)]
    hole = None
    if shaped:
      hole = (
        max(min(it.x0 for it in shaped), tile[0]),
        min(max(it.x1 for it in shaped), tile[1]),
        max(min(it.y0 for it in shaped), tile[2]),
        min(max(it.y1 for it in shaped), tile[3]),
      )
      for x0, x1, y0, y1 in _rects_minus(tile, hole):
        box("floor", x0, x1, y0, y1, -1.0, 0.0, FLOOR_RGBA)
      out.append(self._hfield(spec, body, lay, hole))
    else:
      box("floor", tile[0], tile[1], tile[2], tile[3], -1.0, 0.0, FLOOR_RGBA)
    for it in lay.items:
      if it.cls == "floor":
        if not _shaped(it):
          # A solid down to the slab's base, whatever the item's z0; one on
          # the heightfield sits a hair above it so the two don't z-fight.
          lift = DECAL_M if hole is not None and _overlaps(it, hole) else 0.0
          box("floor", it.x0, it.x1, it.y0, it.y1, -1.0, it.z1 + lift, PLATFORM_RGBA)
        continue
      rgba = WALL_RGBA if it.cls == "wall" else CLASS_RGBA[it.cls]
      box(it.cls, it.x0, it.x1, it.y0, it.y1, it.z0 - EMBED_M, it.z1, rgba)
    return TerrainOutput(origin=np.array([m, m, 0.0]), geometries=out)

  def _hfield(self, spec, body, lay, rect) -> TerrainGeometry:
    """The ground over ``rect`` (layout frame) as a heightfield whose
    vertices sit on ``ground_height``."""
    x0, x1, y0, y1 = rect
    res = self.hfield_resolution
    ncol = round((x1 - x0) / res) + 1
    nrow = round((y1 - y0) / res) + 1
    xs, ys = np.linspace(x0, x1, ncol), np.linspace(y0, y1, nrow)
    gx, gy = np.meshgrid(xs, ys)  # Rows along y, columns along x.
    z = ground_height(lay, gx, gy)
    lo, span = float(z.min()), max(float(z.max() - z.min()), 1e-3)
    field = spec.add_hfield(
      name=f"house_hfield_{next(_names)}",
      size=[(x1 - x0) / 2, (y1 - y0) / 2, span, HFIELD_BASE_M],
      nrow=nrow,
      ncol=ncol,
      userdata=((z - lo) / span).flatten().astype(np.float32).tolist(),
    )
    m = self.margin
    geom = body.add_geom(
      name=f"course_floor_{next(_names)}",
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=(m + (x0 + x1) / 2, m + (y0 + y1) / 2, lo),
    )
    geom.rgba[:] = FLOOR_RGBA
    return TerrainGeometry(geom=geom, hfield=field, color=FLOOR_RGBA)


def house_terrain(name: str, margin: float = 1.0) -> TerrainGeneratorCfg:
  """One tile holding layout ``name``, every robot on it."""
  w, h = LAYOUTS[name].size
  return TerrainGeneratorCfg(
    size=(w + 2 * margin, h + 2 * margin),
    border_width=10.0,
    num_rows=1,
    num_cols=1,
    curriculum=True,
    seed=17,
    sub_terrains={"house": HouseTerrainCfg(layout=name, margin=margin)},
    add_lights=True,
  )


@lru_cache
def route_table(name: str) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
  """Per route of layout ``name``: ``[R, 6]`` start x, y, heading, ground
  height, goal x, y (layout frame), and the global paths ``[L, 2]``."""
  lay = LAYOUTS[name]
  paths = plan_routes(lay)[0]
  rows, kept = [], []
  for (a, b), p in zip(lay.routes, paths, strict=True):
    if p is None:
      raise ValueError(f"{name}: no global path from {a} to {b}.")
    along = np.concatenate([[0.0], np.linalg.norm(np.diff(p, axis=0), axis=1).cumsum()])
    ahead = p[min(int(np.searchsorted(along, HEADING_AHEAD_M)), len(p) - 1)]
    sx, sy = p[0]
    yaw = float(np.arctan2(ahead[1] - sy, ahead[0] - sx))
    z0 = float(ground_height(lay, np.array(sx), np.array(sy)))
    rows.append([sx, sy, yaw, z0, *p[-1]])
    kept.append(np.asarray(p, np.float32))
  return np.array(rows, np.float32), tuple(kept)


HOUSE_TERRAINS = {f"house_{name}": name for name in LAYOUTS}
"""Runner terrain names -> layouts."""
