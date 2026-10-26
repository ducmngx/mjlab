"""A straight corridor with a wall on one side and a ledge on the other.

The hazards are asymmetric: a fault that throws the robot into the wall
(``wall_y`` m left of the centre line, +y) costs a bump, while one that
throws it over the ledge (``ledge_y`` m right, -y) drops it ``drop`` m into
a pit. The goal is straight ahead on the centre line, so a planner blind to
faults walks the middle; one that weighs where a fault would throw it
should keep to the wall side. Each tile is one corridor along +x; the
corridor narrows down the difficulty rows (``ledge_range``), the wall
fixed. The spawn is on the centre line facing +x.

``cliff_waypoints`` gives the goal, relative to the spawn. Geoms are named
like the course's (``course_wall_*``, ``course_floor_*``) so contacts label
the same way.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

from failbench.l_course import FLOOR_RGBA, WALL_RGBA
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

PIT_RGBA = (0.35, 0.30, 0.28, 1.0)

_names = itertools.count()


@dataclass(kw_only=True)
class CliffCourseTerrainCfg(SubTerrainCfg):
  spawn_x: float = 1.0
  wall_y: float = 0.6
  """Centre line to the wall's face (+y)."""
  ledge_range: tuple[float, float] = (0.8, 0.5)
  """Centre line to the ledge's rim (-y) at difficulty 0 and 1."""
  drop: float = 0.6
  """Pit floor below the corridor."""
  goal_x: float = 10.0
  run_out: float = 1.5
  """Corridor past the goal."""
  wall_height: float = 0.8
  wall_thickness: float = 0.15
  floor_depth: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    length, width = self.size
    lo, hi = self.ledge_range
    ledge = lo + (hi - lo) * difficulty
    yc = width / 2
    x_end = self.spawn_x + self.goal_x + self.run_out
    if x_end > length or yc + self.wall_y + self.wall_thickness > width:
      raise ValueError("The corridor doesn't fit the tile; enlarge it.")
    if yc - ledge < 0.5:
      raise ValueError("The pit is under 0.5 m wide; enlarge the tile.")
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

    rim = yc - ledge
    # Corridor floor from the rim to the tile's far edge; the pit below the
    # rest.
    box("floor", 0, length, rim, width, -self.floor_depth, 0, FLOOR_RGBA)
    pit_top = -self.drop
    box("floor", 0, length, 0, rim, pit_top - self.floor_depth, pit_top, PIT_RGBA)
    y0 = yc + self.wall_y
    box(
      "wall",
      0,
      length,
      y0,
      y0 + self.wall_thickness,
      -0.03,
      self.wall_height,
      WALL_RGBA,
    )
    return TerrainOutput(origin=np.array([self.spawn_x, yc, 0.0]), geometries=out)


def cliff_waypoints(cfg: CliffCourseTerrainCfg) -> np.ndarray:
  """The goal (x, y) relative to the spawn origin, as the one waypoint."""
  return np.array([[cfg.goal_x, 0.0]])


CLIFF_COURSE = CliffCourseTerrainCfg(proportion=1.0)
CLIFF_COURSE_TERRAIN = TerrainGeneratorCfg(
  size=(13.0, 3.5),
  border_width=10.0,
  num_rows=10,
  num_cols=1,
  curriculum=True,
  seed=13,
  sub_terrains={"cliff_course": CLIFF_COURSE},
  add_lights=True,
)
"""Ten corridors, the ledge closing in down the rows (0.8 m to 0.5 m)."""
