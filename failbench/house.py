"""Household layouts for the fault-replay evaluation: flats with furniture,
glass, people and level changes, and the routes driven through them.

A layout is a list of ``Item`` boxes in metres, origin at the flat's
lower-left outer corner, z up from the entrance floor. ``floor`` items set
the ground (later ones over earlier ones): raised platforms, sunken pits,
stair treads and ramps; every other class stands on top. Classes are
``failbench/severity.py``'s: ``wall`` 4 (walls, counters), ``box`` 8
(furniture), ``glass`` 25, ``person`` 50.

Stairs use the policy's and the impact model's stair range
(``failbench/env_cfg.py``: 0.08-0.18 m rises, 0.25-0.3 m treads): 0.15 m
rises on 0.30 m treads, below the costmap's 0.18 m wall jump so the
planners see them as walkable; a platform's open edge is a jump the
costmap marks as an obstacle and a drop the critic prices.

``height_map`` rasterises a layout into the planners' elevation grid; the
routes are planned on its Nav2 costmap (``failbench/scripts/plan_house.py``).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch

from failbench.global_planner import plan_paths
from failbench.mppi.nav2 import CostmapCfg, nav2_costmap

WALL_T = 0.15
WALL_Z = 2.0
STEP_RISE = 0.15
STEP_RUN = 0.30
ROUGH_CELL_M = 0.2


@dataclass(frozen=True)
class Item:
  cls: str
  x0: float
  x1: float
  y0: float
  y1: float
  z0: float
  z1: float
  label: str = ""
  ramp: str = ""
  """For a ``floor`` ramp, the direction it rises (``+x``, ``-x``, ``+y``,
  ``-y``) from ``z0`` to ``z1``; empty for a flat top at ``z1``."""
  rough: float = 0.0
  """For a ``floor`` patch, the standard deviation (m) of bumps about its
  top: a smooth random field with features of about ``ROUGH_CELL_M``."""


@dataclass(frozen=True)
class Layout:
  name: str
  size: tuple[float, float]
  items: tuple[Item, ...]
  places: dict[str, tuple[float, float]]
  routes: tuple[tuple[str, str], ...]


def wall_x(x, y0, y1, doors=(), z0=0.0):
  """A wall along y at ``x`` from ``y0`` to ``y1`` with door gaps
  ``(lo, hi)``."""
  out, y = [], y0
  for lo, hi in sorted(doors):
    out.append(Item("wall", x - WALL_T / 2, x + WALL_T / 2, y, lo, z0, WALL_Z))
    y = hi
  out.append(Item("wall", x - WALL_T / 2, x + WALL_T / 2, y, y1, z0, WALL_Z))
  return out


def wall_y(y, x0, x1, doors=(), z0=0.0):
  out, x = [], x0
  for lo, hi in sorted(doors):
    out.append(Item("wall", x, lo, y - WALL_T / 2, y + WALL_T / 2, z0, WALL_Z))
    x = hi
  out.append(Item("wall", x, x1, y - WALL_T / 2, y + WALL_T / 2, z0, WALL_Z))
  return out


def shell(w, h, door_x=None):
  """The outer walls, with the front door in the bottom wall."""
  return [
    *wall_y(0, 0, w, doors=[door_x] if door_x else ()),
    *wall_y(h, 0, w),
    *wall_x(0, 0, h),
    *wall_x(w, 0, h),
  ]


def stairs(x0, x1, y0, y1, rises: str, z_from: float, n: int):
  """``n`` treads filling the rectangle, climbing ``STEP_RISE`` each
  towards ``rises`` from ``z_from`` (the first tread is ``z_from +
  STEP_RISE``); the level beyond the last is one more rise up."""
  out = []
  for i in range(n):
    z = z_from + (i + 1) * STEP_RISE
    if rises in ("+x", "-x"):
      a = x0 + i * STEP_RUN if rises == "+x" else x1 - (i + 1) * STEP_RUN
      out.append(Item("floor", a, a + STEP_RUN, y0, y1, -1.0, z, "stair"))
    else:
      a = y0 + i * STEP_RUN if rises == "+y" else y1 - (i + 1) * STEP_RUN
      out.append(Item("floor", x0, x1, a, a + STEP_RUN, -1.0, z, "stair"))
  return out


def platform(x0, x1, y0, y1, z, label=""):
  return Item("floor", x0, x1, y0, y1, -1.0, z, label)


def on(z, cls, x0, x1, y0, y1, height, label=""):
  """An object of ``height`` standing on ground at ``z``."""
  return Item(cls, x0, x1, y0, y1, z, z + height, label)


# A: a flat apartment (no level changes): living room, kitchen and dining,
# hall with the front door, bedroom.
APARTMENT = Layout(
  name="apartment",
  size=(10.0, 7.0),
  items=(
    *shell(10.0, 7.0, door_x=(8.2, 9.2)),
    *wall_y(4.2, 0, 6, doors=[(2.6, 4.1)]),
    *wall_x(6, 0, 7.0, doors=[(1.4, 2.4), (5.2, 6.2)]),
    *wall_y(3.4, 6, 10.0, doors=[(7.8, 8.8)]),
    on(0, "box", 0.3, 2.4, 0.3, 1.2, 0.8, "sofa"),
    on(0, "glass", 0.9, 1.9, 1.7, 2.3, 0.45, "glass coffee table"),
    on(0, "glass", 0.3, 2.2, 3.75, 4.05, 1.1, "TV"),
    on(0, "box", 4.6, 5.0, 0.3, 0.7, 0.6, "side table"),
    on(0, "glass", 5.4, 5.85, 3.5, 4.05, 1.2, "glass vitrine"),
    on(0, "person", 3.9, 4.3, 1.5, 1.8, 1.7, "person"),
    on(0, "glass", 4.95, 5.3, 2.75, 3.1, 1.6, "floor lamp"),
    on(0, "wall", 0.2, 3.6, 6.3, 6.85, 0.9, "counter"),
    on(0, "box", 3.6, 4.9, 5.0, 5.8, 0.75, "dining table"),
    on(0, "glass", 5.3, 5.85, 6.0, 6.85, 1.8, "glass cabinet"),
    on(0, "box", 0.2, 0.75, 4.4, 5.2, 1.8, "fridge"),
    on(0, "box", 8.2, 9.85, 4.3, 6.4, 0.6, "bed"),
    on(0, "glass", 6.15, 6.45, 3.6, 4.6, 1.6, "mirror"),
    on(0, "box", 6.15, 7.0, 6.3, 6.85, 0.75, "desk"),
    on(0, "box", 9.45, 9.85, 0.4, 1.6, 1.0, "shoe rack"),
    on(0, "glass", 6.9, 7.6, 2.95, 3.3, 0.9, "glass console"),
    # Along the routes.
    on(0, "glass", 6.7, 6.95, 0.25, 0.5, 1.0, "vase"),
    on(0, "person", 8.9, 9.2, 2.0, 2.3, 1.7, "guest"),
    on(0, "box", 7.2, 7.6, 4.0, 4.35, 0.6, "laundry"),
    on(0, "glass", 1.2, 1.6, 4.45, 4.75, 1.0, "wine rack"),
    on(0, "person", 2.4, 2.7, 2.3, 2.6, 1.1, "child"),
    on(0, "box", 5.2, 5.6, 1.0, 1.3, 0.5, "toy box"),
    on(0, "glass", 6.25, 6.35, 0.6, 1.1, 1.7, "standing mirror"),
  ),
  places={
    "front door": (8.7, 0.6),
    "sofa": (3.0, 0.9),
    "TV corner": (1.0, 3.2),
    "desk": (6.6, 5.8),
    "kitchen": (2.0, 5.4),
    "dining": (4.2, 4.6),
    "bedroom": (7.4, 5.2),
    "bed side": (9.0, 3.95),
    "hall": (7.4, 1.0),
  },
  routes=(
    ("front door", "kitchen"),
    ("front door", "sofa"),
    ("sofa", "bedroom"),
    ("kitchen", "bed side"),
    ("TV corner", "hall"),
    ("bedroom", "dining"),
    ("dining", "front door"),
    ("desk", "sofa"),
  ),
)

# B: a split-level loft. The kitchen and dining sit on a platform 0.6 m up
# (four rises), behind a glass balustrade, reached by one open-sided stair
# from the living room; an office and a bedroom on the ground floor.
_UP = 0.6
LOFT = Layout(
  name="loft",
  size=(11.0, 8.0),
  items=(
    platform(0, 7.5, 4.5, 8.0, _UP, "kitchen level +0.6"),
    *stairs(3.0, 4.4, 3.6, 4.5, "+y", 0.0, 3),
    *shell(11.0, 8.0, door_x=(5.5, 6.5)),
    *wall_x(7.5, 0, 8.0, doors=[(1.5, 2.5)]),
    *wall_y(4.5, 7.5, 11.0, doors=[(9.0, 10.0)]),
    on(_UP, "glass", 0.0, 3.0, 4.5, 4.56, 1.0, "glass balustrade"),
    on(_UP, "glass", 4.4, 7.5, 4.5, 4.56, 1.0, "glass balustrade"),
    # Living room, ground floor.
    on(0, "box", 0.3, 0.9, 0.8, 2.8, 0.8, "sofa"),
    on(0, "glass", 1.5, 2.3, 1.4, 2.2, 0.45, "glass table"),
    on(0, "box", 0.3, 2.6, 3.95, 4.45, 0.9, "bookshelf"),
    on(0, "glass", 4.7, 5.2, 3.2, 3.9, 1.4, "display case"),
    on(0, "person", 5.6, 6.0, 2.2, 2.5, 1.7, "person"),
    on(0, "box", 6.6, 7.3, 0.3, 0.8, 0.9, "cabinet"),
    # Kitchen level.
    on(_UP, "wall", 0.2, 4.5, 7.4, 7.85, 0.9, "counter"),
    on(_UP, "box", 2.0, 3.4, 5.4, 6.4, 0.75, "dining table"),
    on(_UP, "glass", 5.8, 6.3, 7.2, 7.85, 1.8, "glass cabinet"),
    on(_UP, "person", 4.6, 5.0, 6.4, 6.7, 1.7, "person"),
    # Office and bedroom, ground floor.
    on(0, "box", 9.8, 10.85, 1.0, 2.6, 0.75, "desk"),
    on(0, "glass", 8.8, 9.9, 2.97, 3.03, 1.6, "glass partition"),
    on(0, "box", 9.0, 10.85, 5.5, 7.6, 0.6, "bed"),
    on(0, "glass", 7.65, 8.0, 6.5, 7.8, 1.8, "mirror"),
    # Along the routes.
    on(0, "glass", 2.3, 2.6, 2.6, 2.9, 1.0, "floor vase"),
    on(0, "box", 4.4, 4.8, 1.1, 1.5, 0.45, "pouf"),
    on(_UP, "person", 5.2, 5.5, 4.65, 4.9, 1.1, "child"),
    on(_UP, "glass", 0.7, 1.0, 5.8, 6.1, 1.2, "plant stand"),
    on(0, "person", 8.6, 8.9, 1.0, 1.3, 1.7, "colleague"),
    on(0, "glass", 7.7, 7.95, 4.7, 4.95, 1.5, "lamp"),
  ),
  places={
    "front door": (6.0, 0.6),
    "sofa": (2.0, 0.6),
    "kitchen": (1.5, 6.8),
    "dining": (4.0, 5.2),
    "pantry": (6.8, 5.5),
    "office": (9.2, 1.8),
    "bedroom": (8.4, 5.6),
  },
  routes=(
    ("front door", "kitchen"),
    ("sofa", "dining"),
    ("kitchen", "office"),
    ("pantry", "front door"),
    ("dining", "bedroom"),
    ("office", "sofa"),
    ("front door", "pantry"),
    ("bedroom", "kitchen"),
  ),
)

# C: a garden behind a house. A paved patio 0.3 m up along the house wall
# (glass patio doors), down one step or a ramp to a lawn of rough patches,
# a grassy berm (about 10 degrees each side), a glass greenhouse, a child
# playing, a shed; a gate in the bottom fence.
_PATIO = 0.3
_BERM = 0.3
GARDEN = Layout(
  name="garden",
  size=(12.0, 9.0),
  items=(
    Item("floor", 0.0, 12.0, 0.0, 6.5, 0.0, 0.0, "lawn", rough=0.025),
    Item("floor", 0.6, 2.4, 0.0, 3.6, 0.0, 0.0, "gravel path"),
    Item("floor", 2.4, 6.0, 0.6, 3.0, 0.0, 0.0, "rough lawn", rough=0.045),
    Item("floor", 6.5, 8.2, 1.0, 3.6, 0.0, _BERM, "berm", ramp="+x"),
    Item("floor", 8.2, 9.0, 1.0, 3.6, 0.0, _BERM, "berm top", rough=0.02),
    Item("floor", 9.0, 10.7, 1.0, 3.6, 0.0, _BERM, "berm", ramp="-x"),
    platform(0.0, 12.0, 6.5, 9.0, _PATIO, "patio +0.3"),
    *stairs(4.6, 6.6, 6.2, 6.5, "+y", 0.0, 1),
    Item("floor", 10.4, 11.8, 4.2, 6.5, 0.0, _PATIO, "ramp", ramp="+y"),
    *wall_y(0, 0, 12.0, doors=[(0.8, 2.2)]),
    *wall_x(0, 0, 9.0),
    *wall_x(12.0, 0, 9.0),
    *wall_y(9.0, 0, 12.0, z0=_PATIO),
    on(_PATIO, "glass", 5.0, 7.2, 8.7, 8.92, 2.0, "glass patio doors"),
    on(0, "box", 0.2, 4.4, 6.0, 6.5, 0.7, "planters"),
    # Patio.
    on(_PATIO, "glass", 1.8, 3.0, 7.3, 8.1, 0.75, "glass table"),
    on(_PATIO, "box", 1.3, 1.7, 7.4, 8.0, 0.9, "chairs"),
    on(_PATIO, "box", 8.6, 9.3, 8.1, 8.75, 1.0, "grill"),
    on(_PATIO, "person", 6.6, 7.0, 8.2, 8.5, 1.7, "person"),
    # Lawn.
    on(0, "glass", 0.4, 2.4, 3.9, 5.6, 2.0, "greenhouse"),
    on(0, "person", 4.6, 4.9, 3.6, 3.9, 1.1, "child"),
    on(0, "box", 3.2, 4.6, 4.8, 5.2, 0.8, "bike"),
    on(0, "glass", 6.6, 7.0, 4.4, 4.8, 0.6, "glass lantern"),
    on(0, "wall", 5.6, 5.9, 0.3, 0.6, 2.0, "tree"),
    on(0, "wall", 9.6, 9.9, 4.6, 4.9, 2.0, "tree"),
    on(0, "box", 10.9, 11.85, 0.2, 2.2, 2.0, "shed"),
    # Along the routes.
    on(0, "glass", 3.4, 3.7, 1.4, 1.7, 0.9, "bird bath"),
    on(0, "person", 5.9, 6.2, 1.5, 1.8, 1.7, "gardener"),
    on(0, "glass", 3.0, 3.8, 3.4, 3.7, 0.4, "cold frame"),
    on(_PATIO, "glass", 5.9, 6.15, 8.1, 8.35, 0.8, "glass planter"),
    on(0, "box", 8.6, 9.6, 4.2, 4.6, 0.8, "bench"),
    on(0, "box", 1.0, 1.5, 2.4, 2.8, 0.6, "wheelbarrow"),
  ),
  places={
    "gate": (1.5, 0.6),
    "patio table": (3.4, 7.2),
    "grill": (8.9, 7.6),
    "greenhouse": (2.9, 4.6),
    "berm top": (8.6, 2.3),
    "shed": (10.3, 0.7),
    "lawn": (7.6, 5.4),
  },
  routes=(
    ("gate", "patio table"),
    ("gate", "grill"),
    ("berm top", "greenhouse"),
    ("greenhouse", "grill"),
    ("patio table", "shed"),
    ("shed", "gate"),
    ("grill", "berm top"),
    ("lawn", "gate"),
  ),
)

# D: a townhouse. A six-rise stair along the right wall climbs to a study
# mezzanine 0.9 m up, behind glass rails; the kitchen sits 0.3 m up a
# gentle ramp (about 7 degrees) or two steps.
_MEZZ = 0.9
_KITCHEN = 0.3
TOWNHOUSE = Layout(
  name="townhouse",
  size=(11.0, 8.0),
  items=(
    platform(7.5, 11.0, 4.2, 8.0, _MEZZ, "study +0.9"),
    *stairs(9.6, 10.85, 2.7, 4.2, "+y", 0.0, 5),
    platform(0.0, 5.0, 5.0, 8.0, _KITCHEN, "kitchen +0.3"),
    Item("floor", 0.2, 1.4, 2.6, 5.0, 0.0, _KITCHEN, "ramp", ramp="+y"),
    *stairs(3.0, 4.4, 4.7, 5.0, "+y", 0.0, 1),
    *shell(11.0, 8.0, door_x=(2.4, 3.4)),
    on(_MEZZ, "glass", 7.5, 9.6, 4.2, 4.26, 1.0, "glass rail"),
    on(_MEZZ, "glass", 7.5, 7.56, 4.2, 8.0, 1.0, "glass rail"),
    on(0, "glass", 9.54, 9.6, 2.7, 4.2, 1.0, "stair rail"),
    on(_KITCHEN, "box", 4.4, 5.0, 5.0, 5.4, 1.0, "bar"),
    on(_KITCHEN, "box", 1.4, 3.0, 5.0, 5.4, 1.0, "bar"),
    # Ground floor living.
    on(0, "box", 4.5, 6.8, 0.3, 1.1, 0.8, "sofa"),
    on(0, "glass", 5.0, 6.2, 1.6, 2.3, 0.45, "glass table"),
    on(0, "glass", 7.2, 7.4, 0.3, 0.5, 1.6, "floor lamp"),
    on(0, "person", 7.6, 8.0, 2.6, 2.9, 1.7, "person"),
    on(0, "box", 0.3, 1.2, 0.3, 1.0, 1.0, "coat stand"),
    on(0, "glass", 5.6, 7.0, 4.25, 4.55, 1.2, "aquarium"),
    # Kitchen.
    on(_KITCHEN, "wall", 0.2, 4.8, 7.4, 7.85, 0.9, "counter"),
    on(_KITCHEN, "box", 2.0, 3.4, 6.0, 6.9, 0.75, "table"),
    on(_KITCHEN, "glass", 4.2, 4.85, 6.0, 6.8, 1.8, "glass cabinet"),
    # Study.
    on(_MEZZ, "box", 9.6, 10.85, 6.6, 7.85, 0.75, "desk"),
    on(_MEZZ, "glass", 7.7, 8.3, 7.3, 7.85, 1.8, "glass bookcase"),
    # Along the routes.
    on(0, "glass", 3.0, 3.3, 1.7, 2.0, 1.0, "vase"),
    on(0, "person", 6.0, 6.3, 3.6, 3.9, 1.1, "child"),
    on(0, "box", 2.4, 2.9, 2.6, 3.0, 0.5, "toy chest"),
    on(_MEZZ, "glass", 10.3, 10.6, 5.1, 5.4, 1.5, "lamp"),
    on(_KITCHEN, "glass", 0.3, 0.55, 5.9, 6.2, 1.0, "vase"),
    on(0, "person", 5.0, 5.3, 3.9, 4.2, 1.7, "guest"),
    on(0, "box", 8.9, 9.2, 2.6, 2.9, 1.0, "plant"),
  ),
  places={
    "front door": (2.9, 0.6),
    "sofa": (5.6, 2.8),
    "kitchen": (1.2, 6.5),
    "table": (3.8, 6.2),
    "study": (8.8, 5.8),
    "desk": (9.2, 6.6),
    "lamp corner": (8.6, 1.0),
  },
  routes=(
    ("front door", "study"),
    ("front door", "kitchen"),
    ("sofa", "desk"),
    ("study", "kitchen"),
    ("table", "lamp corner"),
    ("desk", "front door"),
    ("kitchen", "sofa"),
    ("lamp corner", "study"),
  ),
)

# E: a spacious open-plan lobby. Everything stands back against the walls
# or in seating corners, a metre or more off the walking lines, so a
# careful planner should lose nothing to the normal one here.
LOBBY = Layout(
  name="lobby",
  size=(14.0, 10.0),
  items=(
    *shell(14.0, 10.0, door_x=(6.2, 7.8)),
    on(0, "box", 5.5, 8.5, 8.7, 9.4, 1.1, "reception desk"),
    on(0, "person", 6.8, 7.2, 9.45, 9.75, 1.7, "receptionist"),
    on(0, "box", 0.4, 2.6, 0.3, 1.1, 0.8, "sofa"),
    on(0, "glass", 0.8, 2.0, 1.6, 2.4, 0.45, "glass table"),
    on(0, "box", 11.4, 13.6, 0.3, 1.1, 0.8, "sofa"),
    on(0, "glass", 12.0, 13.2, 1.6, 2.4, 0.45, "glass table"),
    on(0, "glass", 0.25, 0.75, 4.3, 6.3, 1.4, "display case"),
    on(0, "glass", 13.25, 13.75, 4.3, 6.3, 1.4, "display case"),
    on(0, "glass", 0.3, 0.7, 9.3, 9.7, 1.0, "vase"),
    on(0, "glass", 13.3, 13.7, 9.3, 9.7, 1.0, "vase"),
    on(0, "box", 9.6, 12.6, 9.45, 9.85, 1.8, "bookcase"),
    on(0, "box", 1.4, 4.4, 9.45, 9.85, 1.8, "bookcase"),
    on(0, "person", 2.8, 3.1, 0.4, 0.7, 1.7, "visitor"),
    on(0, "person", 12.3, 12.6, 6.8, 7.1, 1.7, "visitor"),
  ),
  places={
    "front door": (7.0, 0.6),
    "reception": (7.0, 7.8),
    "west seats": (2.0, 3.4),
    "east seats": (12.0, 3.4),
    "north-west": (2.6, 8.4),
    "north-east": (11.4, 8.4),
    "west wall": (1.6, 6.8),
    "east wall": (12.4, 5.2),
  },
  routes=(
    ("front door", "reception"),
    ("west seats", "east seats"),
    ("north-west", "east seats"),
    ("north-east", "west seats"),
    ("reception", "front door"),
    ("west wall", "east wall"),
    ("front door", "north-west"),
    ("east seats", "north-east"),
  ),
)

# F: an equally open hall with a group of six people standing in its middle
# (mingling about 1.5 m apart, room to walk between), a buffet and a glass drinks table
# along the top wall. Several routes run straight at the group: the normal
# planner threads it; a careful one should slow or swing wide, and it has
# the room to.
CROWD = Layout(
  name="crowd",
  size=(14.0, 10.0),
  items=(
    *shell(14.0, 10.0, door_x=(0.8, 2.2)),
    on(0, "person", 5.4, 5.8, 3.6, 3.9, 1.7, "guest"),
    on(0, "person", 7.1, 7.5, 3.3, 3.6, 1.7, "guest"),
    on(0, "person", 6.0, 6.4, 5.2, 5.5, 1.7, "guest"),
    on(0, "person", 7.9, 8.3, 5.0, 5.3, 1.7, "guest"),
    on(0, "person", 5.5, 5.9, 6.8, 7.1, 1.7, "guest"),
    on(0, "person", 7.2, 7.6, 6.9, 7.2, 1.7, "guest"),
    on(0, "box", 5.5, 8.5, 9.0, 9.7, 0.9, "buffet"),
    on(0, "glass", 10.4, 11.6, 8.8, 9.6, 0.9, "drinks table"),
    on(0, "box", 0.3, 0.8, 6.0, 8.5, 0.9, "chairs"),
    on(0, "box", 13.2, 13.7, 1.5, 4.0, 0.9, "chairs"),
    on(0, "glass", 13.3, 13.7, 9.3, 9.7, 1.0, "vase"),
  ),
  places={
    "west": (1.0, 5.0),
    "east": (13.0, 5.2),
    "south-west": (1.5, 1.0),
    "north-east": (12.5, 8.4),
    "north-west": (1.5, 9.2),
    "south-east": (12.4, 0.8),
    "south": (7.0, 0.8),
    "buffet": (7.0, 8.4),
  },
  routes=(
    ("west", "east"),
    ("east", "west"),
    ("south-west", "north-east"),
    ("north-west", "south-east"),
    ("south", "buffet"),
    ("buffet", "south"),
    ("north-east", "south-west"),
    ("south-east", "north-west"),
  ),
)

# G: a long hall, open for its first half, then a mingling group of six
# across the walking line. Both planners should drive the open half alike;
# only near the group should a careful one change (here: slow, since the
# group spans the hall's middle).
HALL = Layout(
  name="hall",
  size=(20.0, 8.0),
  items=(
    *shell(20.0, 8.0),
    on(0, "box", 2.0, 4.2, 7.15, 7.85, 0.8, "sofa"),
    on(0, "glass", 6.0, 6.6, 0.15, 0.6, 1.2, "glass cabinet"),
    on(0, "box", 8.0, 9.5, 7.3, 7.85, 0.9, "sideboard"),
    on(0, "person", 11.8, 12.2, 2.8, 3.1, 1.7, "guest"),
    on(0, "person", 12.3, 12.7, 4.6, 4.9, 1.7, "guest"),
    on(0, "person", 13.5, 13.9, 3.5, 3.8, 1.7, "guest"),
    on(0, "person", 13.8, 14.2, 5.2, 5.5, 1.7, "guest"),
    on(0, "person", 15.0, 15.4, 2.7, 3.0, 1.7, "guest"),
    on(0, "person", 15.2, 15.6, 4.4, 4.7, 1.7, "guest"),
    on(0, "glass", 17.0, 18.2, 7.2, 7.85, 0.9, "drinks table"),
  ),
  places={
    "west": (1.0, 4.0),
    "west low": (1.0, 3.2),
    "west high": (1.0, 4.8),
    "west far": (1.0, 2.4),
    "east": (19.0, 4.0),
    "east low": (19.0, 3.2),
    "east high": (19.0, 4.8),
    "east far": (19.0, 5.6),
  },
  routes=(
    ("west", "east"),
    ("east", "west"),
    ("west low", "east high"),
    ("east high", "west low"),
    ("west high", "east low"),
    ("east low", "west high"),
    ("west far", "east far"),
    ("east far", "west far"),
  ),
)

# H: the same long hall with the group to one side: its nearest guests stand
# 0.5 m off the straight walking line, open floor on the other side. A
# careful planner has the room to steer away instead of slowing.
SIDE_CROWD = Layout(
  name="side_crowd",
  size=(20.0, 8.0),
  items=(
    *shell(20.0, 8.0),
    on(0, "box", 2.0, 4.2, 7.15, 7.85, 0.8, "sofa"),
    on(0, "glass", 6.0, 6.6, 0.15, 0.6, 1.2, "glass cabinet"),
    on(0, "person", 9.0, 9.4, 4.5, 4.8, 1.7, "guest"),
    on(0, "person", 10.4, 10.8, 4.6, 4.9, 1.7, "guest"),
    on(0, "person", 11.8, 12.2, 4.5, 4.8, 1.7, "guest"),
    on(0, "person", 9.7, 10.1, 5.9, 6.2, 1.7, "guest"),
    on(0, "person", 11.1, 11.5, 6.0, 6.3, 1.7, "guest"),
    on(0, "person", 12.6, 13.0, 5.8, 6.1, 1.7, "guest"),
    on(0, "glass", 17.0, 18.2, 7.2, 7.85, 0.9, "drinks table"),
  ),
  places={
    "west": (1.0, 4.0),
    "east": (19.0, 4.0),
    "west low": (1.0, 3.4),
    "east low": (19.0, 3.4),
    "west lower": (1.0, 2.6),
    "east lower": (19.0, 2.6),
    "south-west": (1.5, 1.2),
    "south-east": (18.5, 1.2),
  },
  routes=(
    ("west", "east"),
    ("east", "west"),
    ("west low", "east low"),
    ("east low", "west low"),
    ("west", "east low"),
    ("east", "west low"),
    ("west lower", "east"),
    ("east lower", "west"),
  ),
)

LAYOUTS: dict[str, Layout] = {
  lay.name: lay
  for lay in (APARTMENT, LOFT, GARDEN, TOWNHOUSE, LOBBY, CROWD, HALL, SIDE_CROWD)
}


def ground_height(layout: Layout, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
  """The ground's height at points ``(gx, gy)`` (any shape): the ``floor``
  items painted in order (flat tops, ramps, rough bumps), 0 elsewhere. The
  physics (``failbench/house_terrain.py``) and the planners' maps read the
  same surface."""
  ground = np.zeros(np.broadcast_shapes(gx.shape, gy.shape))
  for it in layout.items:
    if it.cls != "floor":
      continue
    m = (gx >= it.x0) & (gx < it.x1) & (gy >= it.y0) & (gy < it.y1)
    if it.ramp:
      axis, lo, hi = (gx, it.x0, it.x1) if "x" in it.ramp else (gy, it.y0, it.y1)
      f = (axis - lo) / (hi - lo)
      f = f if it.ramp[0] == "+" else 1 - f
      ground = np.where(m, it.z0 + f * (it.z1 - it.z0), ground)
    else:
      ground = np.where(m, it.z1, ground)
    if it.rough > 0:
      ground = np.where(m, ground + it.rough * _bumps(gx, gy, it.label), ground)
  return ground


def height_map(
  layout: Layout, resolution: float, pad: float = 0.5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """The layout's elevation grid ``[H, W]`` (top surface: the ground, then
  the tallest object on it; outside the flat a wall), the severity class
  index per cell (-1 ground, else an index into ``CLASSES``), and the cell
  centres' x and y (``[W]``, ``[H]``)."""
  w, h = layout.size
  xs = np.arange(-pad, w + pad, resolution) + resolution / 2
  ys = np.arange(-pad, h + pad, resolution) + resolution / 2
  gx, gy = np.meshgrid(xs, ys)
  top, cls = ground_height(layout, gx, gy), np.full(gx.shape, -1)
  for it in layout.items:
    if it.cls == "floor":
      continue
    m = (gx >= it.x0) & (gx < it.x1) & (gy >= it.y0) & (gy < it.y1) & (it.z1 > top)
    top = np.where(m, it.z1, top)
    cls = np.where(m, CLASSES.index(it.cls), cls)
  outside = (gx < 0) | (gx > w) | (gy < 0) | (gy > h)
  top[outside] = WALL_Z
  cls[outside] = CLASSES.index("wall")
  return top, cls, xs, ys


CLASSES = ("wall", "box", "glass", "person")

_BUMP_EXTENT_M = 20.0


def _bumps(gx: np.ndarray, gy: np.ndarray, seed_text: str) -> np.ndarray:
  """A unit-variance random field at points ``(gx, gy)`` (m, within
  ``_BUMP_EXTENT_M`` of the origin), fixed by ``seed_text``: noise on a
  ``ROUGH_CELL_M`` lattice, interpolated bilinearly (variance of a point
  between lattice nodes is below 1; the lattice's is 1)."""
  rng = np.random.default_rng(zlib.crc32(seed_text.encode()))
  k = round(_BUMP_EXTENT_M / ROUGH_CELL_M) + 2
  c = rng.standard_normal((k, k))
  u = np.clip(gx / ROUGH_CELL_M, 0, k - 1.001)
  v = np.clip(gy / ROUGH_CELL_M, 0, k - 1.001)
  i, j = v.astype(int), u.astype(int)
  fy, fx = v - i, u - j
  return (
    c[i, j] * (1 - fy) * (1 - fx)
    + c[i + 1, j] * fy * (1 - fx)
    + c[i, j + 1] * (1 - fy) * fx
    + c[i + 1, j + 1] * fy * fx
  )


def plan_routes(layout: Layout, resolution: float = 0.05):
  """Each route's global path ``[L, 2]`` (layout frame; None where none is
  found), planned on the Nav2 costmap of the layout's elevation grid
  (``nav2_costmap``: jumps over 0.18 m are obstacles, then inflation) by the
  NavFn-style planner; with the grid, classes, cell centres and costmap."""
  top, cls, xs, ys = height_map(layout, resolution)
  height = torch.as_tensor(top, dtype=torch.float32)[None]
  cost = nav2_costmap(height, resolution, CostmapCfg())[0].numpy()
  centre = np.array([xs.mean(), ys.mean()])
  starts = np.array([layout.places[a] for a, _ in layout.routes]) - centre
  goals = np.array([layout.places[b] for _, b in layout.routes]) - centre
  found = plan_paths(cost, resolution, starts, goals)
  paths = [None if p is None else p + centre for p in found]
  return paths, top, cls, xs, ys, cost
