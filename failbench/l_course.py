"""An L-shaped walled corridor: the robot has to turn.

Each tile is flat ground with box walls forming a corridor that runs
``leg_a`` m along +x from the spawn, turns 90 degrees left at the corner and
runs ``leg_b`` m along +y to the goal, with an end wall ``end_gap`` m past
the goal. The corridor narrows down the difficulty rows
(``half_width_range``), so a fault's drift matters more, most of all at the
corner. The spawn is on the first leg's centre line facing +x.

``corridor_waypoints`` gives the corner and the end, relative to the spawn:
a planner steers to the corner first, so the goal never lies straight
through the inner wall. Geoms are named like the course's
(``course_wall_*``, ``course_floor_*``) so contacts label the same way.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

WALL_RGBA = (0.93, 0.62, 0.15, 1.0)
FLOOR_RGBA = (0.55, 0.62, 0.50, 1.0)
_names = itertools.count()


@dataclass(kw_only=True)
class LCorridorTerrainCfg(SubTerrainCfg):
  spawn_x: float = 1.0
  leg_a: float = 7.0
  """Spawn to the corner along +x (centre line)."""
  leg_b: float = 6.0
  """Corner to the goal along +y (centre line)."""
  end_gap: float = 1.0
  half_width_range: tuple[float, float] = (1.2, 0.8)
  """Corridor half-width at difficulty 0 and 1 (wall face to centre line)."""
  wall_height: float = 0.8
  wall_thickness: float = 0.15
  floor_depth: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    length, width = self.size
    lo, hi = self.half_width_range
    h = lo + (hi - lo) * difficulty
    t = self.wall_thickness
    ya = h + t + 0.5  # first leg's centre line
    xc = self.spawn_x + self.leg_a  # corner (second leg's centre line)
    y_end = ya + self.leg_b + self.end_gap
    if xc + h + t > length or y_end + t > width:
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
    z1 = self.wall_height

    def wall(x0, x1, y0, y1):
      box("wall", x0, x1, y0, y1, -0.03, z1, WALL_RGBA)

    x_back = self.spawn_x - 1.0
    # First leg: outer (right) wall runs to the outer corner, inner (left)
    # wall stops at the second leg's inner face; a cap behind the spawn.
    wall(x_back, xc + h + t, ya - h - t, ya - h)
    wall(x_back, xc - h, ya + h, ya + h + t)
    wall(x_back - t, x_back, ya - h - t, ya + h + t)
    # Second leg: outer wall from the corner, inner wall from the first
    # leg's inner face, and the end wall past the goal.
    wall(xc + h, xc + h + t, ya - h, y_end)
    wall(xc - h - t, xc - h, ya + h, y_end)
    wall(xc - h - t, xc + h + t, y_end, y_end + t)
    return TerrainOutput(origin=np.array([self.spawn_x, ya, 0.0]), geometries=out)


def corridor_waypoints(cfg: LCorridorTerrainCfg) -> np.ndarray:
  """Corner and goal (x, y), relative to the spawn origin."""
  return np.array([[cfg.leg_a, 0.0], [cfg.leg_a, cfg.leg_b]])


L_CORRIDOR = LCorridorTerrainCfg(proportion=1.0)
L_COURSE_TERRAIN = TerrainGeneratorCfg(
  size=(11.0, 11.0),
  border_width=10.0,
  num_rows=10,
  num_cols=1,
  curriculum=True,
  seed=11,
  sub_terrains={"l_corridor": L_CORRIDOR},
  add_lights=True,
)
"""Ten corridors, narrowing down the rows (2.4 m to 1.6 m wide)."""
