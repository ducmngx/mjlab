"""A straight corridor between two identical rows of panels that differ only
in what they are.

The panels left (+y) and right (-y) of the centre line are the same size
and as far away (``side_range``); only their class differs (``classes``,
named ``course_<class>_*``, ``failbench/severity.py``), e.g. cardboard boxes
on one side and glass on the other. A planner that prices only geometry
has no reason to favour a side; one that weighs what a fault would throw
the robot into should keep away from the glass, and swapping the labels
(``SEVERITY_SWAP_TERRAIN``) should mirror its choice. The corridor narrows
down the difficulty rows. The spawn is on the centre line facing +x, the
goal ``goal_x`` ahead on it (``severity_waypoints``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from failbench.l_course import FLOOR_RGBA
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

CLASS_RGBA = {
  "box": (0.72, 0.55, 0.35, 1.0),
  "glass": (0.55, 0.80, 0.95, 0.6),
  "person": (0.90, 0.35, 0.35, 1.0),
  "wall": (0.93, 0.62, 0.15, 1.0),
}

_names = itertools.count()


@dataclass(kw_only=True)
class SeverityCourseTerrainCfg(SubTerrainCfg):
  spawn_x: float = 1.0
  classes: tuple[str, str] = ("box", "glass")
  """The left (+y) and right (-y) panels' class."""
  side_range: tuple[float, float] = (1.0, 0.6)
  """Centre line to either row's face at difficulty 0 and 1."""
  goal_x: float = 10.0
  run_out: float = 1.5
  panel_length: float = 0.9
  """Panels along x, a 0.1 m gap between them (each its own geom)."""
  panel_thickness: float = 0.3
  panel_height: float = 0.5
  floor_depth: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    length, width = self.size
    lo, hi = self.side_range
    side = lo + (hi - lo) * difficulty
    yc = width / 2
    x_end = self.spawn_x + self.goal_x + self.run_out
    if x_end > length or side + self.panel_thickness > yc:
      raise ValueError("The corridor doesn't fit the tile; enlarge it.")
    out = []

    def box(kind, x0, x1, y0, y1, z0, z1, rgba):
      geom = body.add_geom(
        name=f"course_{kind}_{next(_names)}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
        pos=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2),
      )
      geom.rgba[:] = rgba
      out.append(TerrainGeometry(geom=geom, color=rgba))

    box("floor", 0, length, 0, width, -self.floor_depth, 0, FLOOR_RGBA)
    t, z1 = self.panel_thickness, self.panel_height
    pitch = self.panel_length + 0.1
    for x0 in np.arange(0.0, x_end, pitch):
      x1 = min(x0 + self.panel_length, x_end)
      for cls, y0, y1 in (
        (self.classes[0], yc + side, yc + side + t),
        (self.classes[1], yc - side - t, yc - side),
      ):
        box(cls, x0, x1, y0, y1, -0.03, z1, CLASS_RGBA.get(cls, CLASS_RGBA["wall"]))
    return TerrainOutput(origin=np.array([self.spawn_x, yc, 0.0]), geometries=out)


def severity_waypoints(cfg: SeverityCourseTerrainCfg) -> np.ndarray:
  """The goal (x, y) relative to the spawn origin, as the one waypoint."""
  return np.array([[cfg.goal_x, 0.0]])


def _terrain(cfg: SeverityCourseTerrainCfg) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(13.0, 3.5),
    border_width=10.0,
    num_rows=10,
    num_cols=1,
    curriculum=True,
    seed=17,
    sub_terrains={"severity_course": cfg},
    add_lights=True,
  )


SEVERITY_COURSE = SeverityCourseTerrainCfg(proportion=1.0)
SEVERITY_COURSE_TERRAIN = _terrain(SEVERITY_COURSE)
"""Boxes left, glass right; the corridor narrowing from 2 m to 1.2 m."""
SEVERITY_SWAP = replace(SEVERITY_COURSE, classes=SEVERITY_COURSE.classes[::-1])
SEVERITY_SWAP_TERRAIN = _terrain(SEVERITY_SWAP)
"""The same geometry, glass left and boxes right."""
