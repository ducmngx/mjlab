"""Walled rooms where only what the objects are tells the planner how close
to pass them.

Scenes, each with a twin that swaps the classes at the same places
(``swap``), so a planner that prices only geometry drives both alike and
one that weighs what a fault would throw the robot into mirrors itself:

- ``ClusterCourseTerrainCfg``: crates (``box``) and glass display cases
  (``glass``) off either side of the straight line to the goal, each side
  holding both; the line itself is clear.
- ``GapCourseTerrainCfg``: a wall across the room with one opening on the
  straight line, glass panes on one side of it and crates on the other.
- ``PassAroundCourseTerrainCfg``: a pillar on the line to go round, glass
  beyond one side and crates beyond the other.
- ``ClutterFieldTerrainCfg``: a fresh random field of crates and glass
  cases per tile, either side of a clear corridor along the line.

The objects stand where the robot is when the fault fires (4-8 s into the
run, ``run_mppi``: 1.5-5 m out). Their faces close in on the line down the
difficulty rows. Geoms are named ``course_<class>_*`` (``failbench/
severity.py``). The spawn is on the centre line facing +x, the goal
``goal_x`` ahead on it (``scene_waypoints``).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from failbench.l_course import FLOOR_RGBA
from failbench.severity_course import CLASS_RGBA
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeneratorCfg,
  TerrainGeometry,
  TerrainOutput,
)

WALL_RGBA = (0.55, 0.54, 0.52, 1.0)
STAND_RGBA = (0.62, 0.66, 0.70, 1.0)
SWAP = {"box": "glass", "glass": "box"}

_names = itertools.count()


@dataclass(kw_only=True)
class _RoomCfg(SubTerrainCfg):
  spawn_x: float = 1.0
  goal_x: float = 8.0
  run_out: float = 1.5
  half_width: float = 2.0
  """Centre line to the side walls' face."""
  wall_height: float = 0.6
  wall_thickness: float = 0.2
  floor_depth: float = 1.0
  swap: bool = False
  """Swap glass and crates, keeping every place."""

  def _room(self, spec: mujoco.MjSpec):
    body = spec.body("terrain")
    length, width = self.size
    yc = width / 2
    x_end = self.spawn_x + self.goal_x + self.run_out
    if x_end > length or self.half_width + self.wall_thickness > yc:
      raise ValueError("The room doesn't fit the tile; enlarge it.")
    out = []

    def box(cls, x0, x1, y0, y1, z0, z1, rgba=None):
      """In spawn coordinates: x from the spawn, y from the centre line."""
      geom = body.add_geom(
        name=f"course_{cls}_{next(_names)}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2),
        pos=(
          self.spawn_x + (x0 + x1) / 2,
          yc + (y0 + y1) / 2,
          (z0 + z1) / 2,
        ),
      )
      color = rgba or (WALL_RGBA if cls == "wall" else CLASS_RGBA[cls])
      geom.rgba[:] = color
      out.append(TerrainGeometry(geom=geom, color=color))

    floor = (-self.spawn_x, length - self.spawn_x, -yc, yc, -self.floor_depth, 0)
    box("floor", *floor, FLOOR_RGBA)
    hw, t, h = self.half_width, self.wall_thickness, self.wall_height
    x_lo, x_hi = -self.spawn_x, x_end - self.spawn_x
    box("wall", x_lo, x_hi, hw, hw + t, -0.03, h)
    box("wall", x_lo, x_hi, -hw - t, -hw, -0.03, h)
    box("wall", x_hi - t, x_hi, -hw, hw, -0.03, h)
    return box, out, np.array([self.spawn_x, yc, 0.0])

  def _cls(self, cls: str) -> str:
    return SWAP.get(cls, cls) if self.swap else cls


def crates(box, x: float, y: float, size: tuple[float, float]) -> None:
  """A pile of crates filling ``size`` (x, y) about (x, y): two side by side,
  a third on top."""
  sx, sy = size
  for i in (0, 1):
    x0 = x - sx / 2 + i * sx / 2
    box("box", x0 + 0.02, x0 + sx / 2 - 0.02, y - sy / 2, y + sy / 2, -0.03, 0.42)
  box("box", x - sx / 4, x + sx / 4, y - sy / 2 + 0.05, y + sy / 2 - 0.05, 0.42, 0.8)


def glass_case(box, x: float, y: float, size: tuple[float, float]) -> None:
  """A display case filling ``size``: a stand with a glass box on it, both
  glass to the planner (hitting either breaks it)."""
  sx, sy = size
  x0, x1, y0, y1 = x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2
  box("glass", x0, x1, y0, y1, -0.03, 0.35, STAND_RGBA)
  box("glass", x0, x1, y0, y1, 0.35, 1.1)


OBJECTS = {"box": crates, "glass": glass_case}


@dataclass(kw_only=True)
class ClusterCourseTerrainCfg(_RoomCfg):
  layout: tuple[tuple[str, float, int], ...] = (
    ("box", 1.8, 1),
    ("glass", 3.0, -1),
    ("glass", 4.4, 1),
    ("box", 5.6, -1),
  )
  """(class, x from the spawn, side: +1 left, -1 right) per object."""
  object_size: tuple[float, float] = (0.8, 0.5)
  face_range: tuple[float, float] = (0.6, 0.4)
  """Centre line to the objects' near face at difficulty 0 and 1. The Go2
  is about 0.31 m wide; a limp hip throws it 0.4-0.6 m by 2 s."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    box, out, origin = self._room(spec)
    lo, hi = self.face_range
    face = lo + (hi - lo) * difficulty
    for cls, x, side in self.layout:
      y = side * (face + self.object_size[1] / 2)
      OBJECTS[self._cls(cls)](box, x, y, self.object_size)
    return TerrainOutput(origin=origin, geometries=out)


@dataclass(kw_only=True)
class GapCourseTerrainCfg(_RoomCfg):
  gap_x: float = 3.5
  """The cross wall's near face, from the spawn."""
  depth: float = 0.3
  jamb: float = 0.8
  """Width of the glass and the crates either side of the opening."""
  gap_range: tuple[float, float] = (1.3, 0.9)
  """Opening width at difficulty 0 and 1."""
  sides: tuple[str, str] = ("glass", "box")
  """Left (+y) and right (-y) of the opening."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    box, out, origin = self._room(spec)
    lo, hi = self.gap_range
    half = (lo + (hi - lo) * difficulty) / 2
    x0, x1 = self.gap_x, self.gap_x + self.depth
    hw = self.half_width
    for s, cls in zip((1, -1), self.sides, strict=True):
      a, b = sorted((s * half, s * (half + self.jamb)))
      cls = self._cls(cls)
      if cls == "glass":
        box("glass", x0, x1, a, b, -0.03, 0.9)
      else:
        crates(box, (x0 + x1) / 2, (a + b) / 2, (self.depth, b - a))
      a, b = sorted((s * (half + self.jamb), s * hw))
      box("wall", x0, x1, a, b, -0.03, self.wall_height)
    return TerrainOutput(origin=origin, geometries=out)


@dataclass(kw_only=True)
class PassAroundCourseTerrainCfg(_RoomCfg):
  """A pillar on the line, glass beyond its left, crates beyond its right:
  the robot must go round and should take the crate side."""

  pillar_x: float = 3.5
  pillar: float = 0.6
  """The pillar's side (a plain wall block)."""
  side_face_range: tuple[float, float] = (1.3, 1.0)
  """Centre line to the side objects' near face at difficulty 0 and 1: a
  1.0 m passage either side of the pillar, closing to 0.7 m (at 0.5 m the
  footprint and its spread can't see a way through)."""
  object_size: tuple[float, float] = (1.0, 0.5)

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    box, out, origin = self._room(spec)
    p, x = self.pillar / 2, self.pillar_x
    box("wall", x - p, x + p, -p, p, -0.03, self.wall_height)
    lo, hi = self.side_face_range
    face = lo + (hi - lo) * difficulty
    for side, cls in ((1, "glass"), (-1, "box")):
      y = side * (face + self.object_size[1] / 2)
      OBJECTS[self._cls(cls)](box, x, y, self.object_size)
    return TerrainOutput(origin=origin, geometries=out)


@dataclass(kw_only=True)
class ClutterFieldTerrainCfg(_RoomCfg):
  """Crates and glass cases scattered either side of a clear corridor on the
  line, a new layout per tile from the generator's rng. The draws don't
  depend on ``swap``, so a twin built from the same seed has every object
  where it was, its class swapped."""

  count_range: tuple[int, int] = (6, 14)
  """Objects drawn per tile (inclusive), half of them glass; one that won't
  fit after ``tries`` draws is left out."""
  corridor_range: tuple[float, float] = (1.6, 0.8)
  """Free width about the line at difficulty 0 and 1."""
  x_range: tuple[float, float] = (1.0, 7.0)
  """Object centres from the spawn: around where faults fire (1.5-6 m)."""
  size_x_range: tuple[float, float] = (0.4, 1.0)
  size_y_range: tuple[float, float] = (0.3, 0.6)
  set_back_range: tuple[float, float] = (0.0, 0.5)
  """Corridor edge to an object's near face."""
  spacing: float = 0.1
  """Least gap between objects."""
  tries: int = 30

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    box, out, origin = self._room(spec)
    lo, hi = self.corridor_range
    half = (lo + (hi - lo) * difficulty) / 2
    n = int(rng.integers(self.count_range[0], self.count_range[1] + 1))
    classes = rng.permutation(["glass"] * (n // 2) + ["box"] * (n - n // 2))
    placed: list[tuple[float, float, float, float]] = []  # x0, x1, y0, y1
    for cls in classes:
      for _ in range(self.tries):
        sx, sy = rng.uniform(*self.size_x_range), rng.uniform(*self.size_y_range)
        x = rng.uniform(*self.x_range)
        side = 1 if rng.random() < 0.5 else -1
        face = half + rng.uniform(*self.set_back_range)
        y = side * (face + sy / 2)
        x0, x1, y0, y1 = x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2
        g = self.spacing
        fits = face + sy <= self.half_width - g and not any(
          x0 < b1 + g and b0 < x1 + g and y0 < c1 + g and c0 < y1 + g
          for b0, b1, c0, c1 in placed
        )
        if fits:
          placed.append((x0, x1, y0, y1))
          OBJECTS[self._cls(str(cls))](box, x, y, (sx, sy))
          break
    return TerrainOutput(origin=origin, geometries=out)


def scene_waypoints(cfg: _RoomCfg) -> np.ndarray:
  """The goal (x, y) relative to the spawn origin, as the one waypoint."""
  return np.array([[cfg.goal_x, 0.0]])


def _terrain(cfg: _RoomCfg, num_cols: int = 1) -> TerrainGeneratorCfg:
  """One difficulty per row; ``num_cols`` copies of the scene side by side
  (curriculum mode gives each sub-terrain one column), which differ only
  if the scene draws from the rng."""
  return TerrainGeneratorCfg(
    size=(11.0, 4.8),
    border_width=10.0,
    num_rows=10,
    num_cols=num_cols,
    curriculum=True,
    seed=17,
    sub_terrains={f"scene_{i}": cfg for i in range(num_cols)},
    add_lights=True,
  )


CLUSTER_COURSE = ClusterCourseTerrainCfg(proportion=1.0)
CLUSTER_COURSE_TERRAIN = _terrain(CLUSTER_COURSE)
"""Crates and glass cases either side; faces 0.6 m off the line, closing to
0.4 m."""
CLUSTER_SWAP = replace(CLUSTER_COURSE, swap=True)
CLUSTER_SWAP_TERRAIN = _terrain(CLUSTER_SWAP)
GAP_COURSE = GapCourseTerrainCfg(proportion=1.0)
GAP_COURSE_TERRAIN = _terrain(GAP_COURSE)
"""Glass left of the opening, crates right; 1.3 m wide, narrowing to 0.9 m."""
GAP_SWAP = replace(GAP_COURSE, swap=True)
GAP_SWAP_TERRAIN = _terrain(GAP_SWAP)
PASS_COURSE = PassAroundCourseTerrainCfg(proportion=1.0)
PASS_COURSE_TERRAIN = _terrain(PASS_COURSE)
"""A pillar on the line; glass beyond its left, crates beyond its right."""
PASS_SWAP = replace(PASS_COURSE, swap=True)
PASS_SWAP_TERRAIN = _terrain(PASS_SWAP)
CLUTTER_FIELD = ClutterFieldTerrainCfg(proportion=1.0)
CLUTTER_FIELD_TERRAIN = _terrain(CLUTTER_FIELD, num_cols=4)
"""40 random fields, 1.6 m corridors narrowing to 0.8 m down the rows."""
CLUTTER_SWAP = replace(CLUTTER_FIELD, swap=True)
CLUTTER_SWAP_TERRAIN = _terrain(CLUTTER_SWAP, num_cols=4)
