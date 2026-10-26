"""A straight walled lane with narrow doorways and a goal in front of a wall.

A test of how far a fault's drift carries the robot before feedback can pull
it back, against how much room it has: each tile is a lane ``half_width`` m
either side of its centre line, crossed by ``door_x`` walls, each with one
doorway ``gap`` m wide whose centre alternates ``door_offset`` m either side
of the centre line, so the path zigzags and never runs straight through.
The goal sits ``end_gap`` m in front of the lane's end wall, so a robot that
can't stop in time hits it. The doorways narrow down the difficulty rows
(``gap_range``). The spawn is on the centre line facing +x.

``door_waypoints`` gives points in front of and just past each doorway,
then the goal, relative to the spawn. Geoms are named like the course's (``course_wall_*``,
``course_floor_*``) so contacts label the same way.
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

_names = itertools.count()


@dataclass(kw_only=True)
class DoorCourseTerrainCfg(SubTerrainCfg):
  spawn_x: float = 1.0
  half_width: float = 1.0
  """Lane wall face to centre line."""
  door_x: tuple[float, ...] = (6.0, 9.0)
  """Door walls' centres, from the spawn along +x. The fault fires at 4-8 s
  (``run_mppi``), after 1.5-5 m: before or around the first door."""
  door_offset: float = 0.45
  """Doorway centre off the centre line: left at the first door, then
  alternating."""
  door_depth: float = 0.2
  gap_range: tuple[float, float] = (1.0, 0.6)
  """Doorway width at difficulty 0 and 1. The Go2 is about 0.31 m wide."""
  goal_x: float = 12.0
  end_gap: float = 0.6
  """Goal to the end wall's face."""
  wall_height: float = 0.8
  wall_thickness: float = 0.15
  floor_depth: float = 1.0

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    length, width = self.size
    lo, hi = self.gap_range
    gap = lo + (hi - lo) * difficulty
    h, t = self.half_width, self.wall_thickness
    yc = width / 2
    x_end = self.spawn_x + self.goal_x + self.end_gap
    if x_end + t > length or 2 * (h + t) > width:
      raise ValueError("The lane doesn't fit the tile; enlarge it.")
    if self.door_offset + gap / 2 > h:
      raise ValueError("A doorway runs into the lane wall.")
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

    def wall(x0, x1, y0, y1):
      box("wall", x0, x1, y0, y1, -0.03, self.wall_height, WALL_RGBA)

    x_back = self.spawn_x - 1.0
    # Lane walls, the cap behind the spawn and the end wall past the goal.
    wall(x_back, x_end, yc - h - t, yc - h)
    wall(x_back, x_end, yc + h, yc + h + t)
    wall(x_back - t, x_back, yc - h - t, yc + h + t)
    wall(x_end, x_end + t, yc - h - t, yc + h + t)
    # Door walls: the lane closed but for one doorway each.
    for i, dx in enumerate(self.door_x):
      x0 = self.spawn_x + dx - self.door_depth / 2
      x1 = x0 + self.door_depth
      yd = yc + self.door_offset * (1 if i % 2 == 0 else -1)
      wall(x0, x1, yc - h, yd - gap / 2)
      wall(x0, x1, yd + gap / 2, yc + h)
    return TerrainOutput(origin=np.array([self.spawn_x, yc, 0.0]), geometries=out)


def door_waypoints(
  cfg: DoorCourseTerrainCfg, before: float = 0.8, past: float = 0.5
) -> np.ndarray:
  """Points ``before`` m ahead of and ``past`` m beyond each doorway's
  centre, then the goal (x, y), relative to the spawn origin: the robot
  lines up with a doorway before going through it, rather than cutting
  across its edge."""
  doors = []
  for i, dx in enumerate(cfg.door_x):
    yd = cfg.door_offset * (1 if i % 2 == 0 else -1)
    doors += [[dx - before, yd], [dx + past, yd]]
  return np.array([*doors, [cfg.goal_x, 0.0]])


DOOR_COURSE = DoorCourseTerrainCfg(proportion=1.0)
DOOR_COURSE_TERRAIN = TerrainGeneratorCfg(
  size=(14.5, 3.5),
  border_width=10.0,
  num_rows=10,
  num_cols=1,
  curriculum=True,
  seed=11,
  sub_terrains={"door_course": DOOR_COURSE},
  add_lights=True,
)
"""Ten lanes, the doorways narrowing down the rows (1.0 m to 0.6 m)."""
