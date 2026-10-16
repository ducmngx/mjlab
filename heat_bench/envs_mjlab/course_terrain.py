"""Obstacle-lane tiles for fault impact trials.

Each tile is a lane along +x down the middle of the tile. The robot spawns at
the lane start facing +x, and a healthy robot walking straight never touches
anything: obstacles (walls, pillars, pole fences, crates) stand beside the
lane, and the floor feature (rough ground, a ramp or a flight of stairs)
spans the full tile width. Contacts with obstacles then come from the drift a
joint fault causes, not from the command.

Everything is a box, cylinder or heightfield: the scene only needs to be
representative of what a drifting or falling robot can hit, not realistic.
One difficulty knob (the curriculum row) makes the floor feature more severe
and brings more obstacles closer to the lane.

Obstacle geoms are named ``course_{kind}_{n}`` and floor geoms
``course_floor_{n}``, so a contact can later be labelled by what it hit.
"""

from __future__ import annotations

import itertools
import math
import uuid
from dataclasses import dataclass, replace
from typing import Callable, Literal

import mujoco
import numpy as np
from scipy import ndimage

from mjlab.terrains.heightfield_terrains import color_by_height
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)

Floor = Literal["flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down"]
FLOORS: tuple[Floor, ...] = (
  "flat",
  "rough",
  "slope_up",
  "slope_down",
  "stairs_up",
  "stairs_down",
)
FloorHeight = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Floor height at tile-frame points ``(x, y)``."""

OBSTACLE_KINDS = ("wall", "pillar", "pole_fence", "crate")

FLOOR_RGBA = (0.55, 0.62, 0.50, 1.0)
OBSTACLE_RGBA = {
  "wall": (0.93, 0.62, 0.15, 1.0),
  "pillar": (0.80, 0.20, 0.20, 1.0),
  "pole_fence": (0.20, 0.35, 0.80, 1.0),
  "crate": (0.70, 0.55, 0.35, 1.0),
}

# Floor boxes extend this far below their top surface so feet never find an
# edge to slip under.
_FLOOR_DEPTH = 1.0
# Obstacles start this far below the lowest floor under their footprint, so
# they sit flush on slopes, stairs and rough ground.
_EMBED = 0.03
# How far a flat floor piece reaches under a higher neighbour.
_SEAM_OVERLAP = 0.01
# Gap kept between obstacle footprints.
_OBSTACLE_GAP = 0.15

_names = itertools.count()


def _name(kind: str) -> str:
  return f"course_{kind}_{next(_names)}"


def _lerp(lo_hi: tuple[float, float], t: float) -> float:
  return lo_hi[0] + (lo_hi[1] - lo_hi[0]) * t


@dataclass(frozen=True)
class FloorPiece:
  """A floor section over ``[x0, x1)`` whose height runs linearly from
  ``z0`` to ``z1`` (equal for flat pieces and stair treads)."""

  x0: float
  x1: float
  z0: float
  z1: float


def floor_profile(
  floor: Floor,
  difficulty: float,
  length: float,
  feature_start: float,
  slope_deg_range: tuple[float, float],
  ramp_length: float,
  riser_range: tuple[float, float],
  tread: float,
  num_steps_range: tuple[int, int],
) -> list[FloorPiece]:
  """Height profile along the lane for every floor except ``rough``.

  Stairs and slopes start at ``feature_start``; going down starts raised so
  the robot always spawns on flat ground.
  """
  if floor == "flat":
    return [FloorPiece(0.0, length, 0.0, 0.0)]
  if floor in ("slope_up", "slope_down"):
    angle = math.radians(_lerp(slope_deg_range, difficulty))
    rise = ramp_length * math.tan(angle)
    x1 = feature_start + ramp_length
    lo, hi = (0.0, rise) if floor == "slope_up" else (rise, 0.0)
    return [
      FloorPiece(0.0, feature_start, lo, lo),
      FloorPiece(feature_start, x1, lo, hi),
      FloorPiece(x1, length, hi, hi),
    ]
  if floor in ("stairs_up", "stairs_down"):
    riser = _lerp(riser_range, difficulty)
    num_steps = int(round(_lerp(num_steps_range, difficulty)))
    sign = 1.0 if floor == "stairs_up" else -1.0
    z = 0.0 if floor == "stairs_up" else riser * num_steps
    pieces = [FloorPiece(0.0, feature_start, z, z)]
    x = feature_start
    for _ in range(num_steps):
      z += sign * riser
      pieces.append(FloorPiece(x, x + tread, z, z))
      x += tread
    pieces.append(FloorPiece(x, length, z, z))
    return pieces
  raise ValueError(f"No profile for floor {floor!r}.")


def profile_height(pieces: list[FloorPiece], x: np.ndarray) -> np.ndarray:
  """Floor height at lane positions ``x`` (clamped to the profile ends)."""
  x = np.asarray(x, dtype=float)
  z = np.full(x.shape, pieces[0].z0)
  for p in pieces:
    inside = (x >= p.x0) & (x < p.x1)
    t = (x[inside] - p.x0) / max(p.x1 - p.x0, 1e-9)
    z[inside] = p.z0 + t * (p.z1 - p.z0)
  z[x >= pieces[-1].x1] = pieces[-1].z1
  return z


@dataclass(kw_only=True)
class ObstacleLaneTerrainCfg(SubTerrainCfg):
  """A lane along +x with one floor feature and obstacles beside it.

  ``size`` is set by the generator. Difficulty ``d`` in [0, 1] interpolates
  every ``*_range`` from its first to its second value.
  """

  floor: Floor = "flat"
  spawn_x: float = 1.0
  """Spawn origin distance from the tile's -x edge, on the lane centre."""
  feature_start: float = 3.0
  """Where the ramp or stairs begin. Fault times of 3-5 s at walking speed
  put many faults on or just before the feature."""

  slope_deg_range: tuple[float, float] = (5.0, 20.0)
  ramp_length: float = 3.0
  riser_range: tuple[float, float] = (0.03, 0.14)
  """Stair riser height. The reference policy trained on risers up to
  ~0.10 m, so the hardest rows are beyond what it has seen."""
  tread: float = 0.3
  num_steps_range: tuple[int, int] = (5, 8)
  rough_amplitude_range: tuple[float, float] = (0.02, 0.10)
  """Peak-to-peak height of the rough floor."""
  rough_feature_m: float = 0.6
  """Rough-floor bump size (Gaussian smoothing length)."""
  rough_resolution: float = 0.05

  num_obstacles_range: tuple[int, int] = (3, 8)
  near_clearance_range: tuple[float, float] = (1.0, 0.55)
  """Closest an obstacle's inner face gets to the lane centre. Kept above
  0.5 m so obstacles stay outside the policy's +-0.5 m height scan while
  the robot walks straight."""
  far_clearance_range: tuple[float, float] = (2.0, 1.0)
  obstacle_x_margin: tuple[float, float] = (2.0, 1.0)
  """Obstacle-free length at the lane start (spawn area) and end."""
  kind_weights: tuple[float, float, float, float] = (0.3, 0.25, 0.15, 0.3)
  """Sampling weights for wall, pillar, pole_fence, crate."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    length, width = self.size
    y_mid = width / 2
    geometries: list[TerrainGeometry] = []

    if self.floor == "rough":
      heights, floor_geoms = self._rough_floor(body, spec, difficulty, rng)

      def floor_at(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        res = self.rough_resolution
        ix = np.clip(np.round(xs / res).astype(int), 0, heights.shape[1] - 1)
        iy = np.clip(np.round(ys / res).astype(int), 0, heights.shape[0] - 1)
        return heights[iy, ix]
    else:
      pieces = floor_profile(
        self.floor,
        difficulty,
        length,
        self.feature_start,
        self.slope_deg_range,
        self.ramp_length,
        self.riser_range,
        self.tread,
        self.num_steps_range,
      )
      floor_geoms = _floor_boxes(body, pieces, width)

      def floor_at(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        del ys
        return profile_height(pieces, xs)

    geometries += floor_geoms
    geometries += self._obstacles(body, difficulty, rng, floor_at)

    spawn_z = float(floor_at(np.array([self.spawn_x]), np.array([y_mid]))[0])
    origin = np.array([self.spawn_x, y_mid, spawn_z])
    return TerrainOutput(origin=origin, geometries=geometries)

  def _rough_floor(
    self,
    body: mujoco.MjsBody,
    spec: mujoco.MjSpec,
    difficulty: float,
    rng: np.random.Generator,
  ) -> tuple[np.ndarray, list[TerrainGeometry]]:
    """Smoothed-noise heightfield over the whole tile, flat at the spawn.

    Returns heights ``[ny, nx]`` (row = y) and the geometry."""
    length, width = self.size
    res = self.rough_resolution
    nx, ny = int(round(length / res)) + 1, int(round(width / res)) + 1
    noise = ndimage.gaussian_filter(
      rng.uniform(-1.0, 1.0, (ny, nx)), sigma=self.rough_feature_m / res / 2
    )
    noise += 0.15 * rng.uniform(-1.0, 1.0, (ny, nx))  # Small-scale grit.
    noise = (noise - noise.min()) / max(float(np.ptp(noise)), 1e-9)
    # Fade in over a metre after the spawn pad, so the robot starts flat.
    xs = np.arange(nx) * res
    fade = np.clip((xs - (self.spawn_x + 0.5)) / 1.0, 0.0, 1.0)
    normalized = (noise * fade[None, :]).astype(np.float32)

    amplitude = max(_lerp(self.rough_amplitude_range, difficulty), 1e-3)
    uid = uuid.uuid4().hex
    field = spec.add_hfield(
      name=f"course_hfield_{uid}",
      size=[length / 2, width / 2, amplitude, _FLOOR_DEPTH],
      nrow=ny,
      ncol=nx,
      userdata=normalized.flatten().tolist(),
    )
    heights = normalized * amplitude
    material = color_by_height(spec, normalized, uid, heights)
    geom = body.add_geom(
      name=_name("floor"),
      type=mujoco.mjtGeom.mjGEOM_HFIELD,
      hfieldname=field.name,
      pos=[length / 2, width / 2, 0.0],
      material=material,
    )
    return heights, [TerrainGeometry(geom=geom, hfield=field)]

  def _obstacles(
    self,
    body: mujoco.MjsBody,
    difficulty: float,
    rng: np.random.Generator,
    floor_at: FloorHeight,
  ) -> list[TerrainGeometry]:
    length, width = self.size
    y_mid = width / 2
    near = _lerp(self.near_clearance_range, difficulty)
    far = max(_lerp(self.far_clearance_range, difficulty), near)
    count = int(round(_lerp(self.num_obstacles_range, difficulty)))
    weights = np.asarray(self.kind_weights, dtype=float)
    weights /= weights.sum()
    x_lo, x_hi = self.obstacle_x_margin[0], length - self.obstacle_x_margin[1]

    placed: list[tuple[float, float, float, float]] = []  # AABBs.
    geometries: list[TerrainGeometry] = []
    for _ in range(count):
      kind = OBSTACLE_KINDS[int(rng.choice(len(OBSTACLE_KINDS), p=weights))]
      for _attempt in range(20):
        parts = _sample_obstacle(kind, rng)
        half_x, half_y = _extent(parts)
        side = 1.0 if rng.uniform() < 0.5 else -1.0
        clearance = rng.uniform(near, far)
        cx = rng.uniform(x_lo + half_x, x_hi - half_x)
        cy = y_mid + side * (clearance + half_y)
        if abs(cy - y_mid) + half_y > width / 2 - 0.1:
          continue  # Would stick out of the tile.
        box = (cx - half_x, cx + half_x, cy - half_y, cy + half_y)
        if any(_overlaps(box, other, _OBSTACLE_GAP) for other in placed):
          continue
        placed.append(box)
        geometries += _add_obstacle(body, kind, parts, cx, cy, floor_at)
        break
    return geometries


@dataclass(frozen=True)
class _Part:
  """One primitive of an obstacle, relative to the obstacle centre.
  ``half`` is (hx, hy) for boxes and (r, r) for cylinders."""

  shape: Literal["box", "cylinder"]
  dx: float
  half: tuple[float, float]
  height: float
  yaw: float = 0.0


def _sample_obstacle(kind: str, rng: np.random.Generator) -> list[_Part]:
  if kind == "wall":
    return [_Part("box", 0.0, (rng.uniform(0.5, 1.25), 0.075), rng.uniform(0.6, 1.2))]
  if kind == "pillar":
    r = rng.uniform(0.10, 0.25)
    return [_Part("cylinder", 0.0, (r, r), rng.uniform(0.8, 1.5))]
  if kind == "pole_fence":
    n = int(rng.integers(4, 9))
    spacing = rng.uniform(0.15, 0.3)
    span = (n - 1) * spacing
    return [
      _Part("cylinder", -span / 2 + i * spacing, (0.05, 0.05), 1.0) for i in range(n)
    ]
  if kind == "crate":
    return [
      _Part(
        "box",
        0.0,
        (rng.uniform(0.15, 0.4), rng.uniform(0.15, 0.4)),
        rng.uniform(0.2, 0.6),
        yaw=rng.uniform(0.0, math.pi / 2),
      )
    ]
  raise ValueError(f"Unknown obstacle kind {kind!r}.")


def _part_extent(p: _Part) -> tuple[float, float]:
  hx, hy = p.half
  c, s = abs(math.cos(p.yaw)), abs(math.sin(p.yaw))
  return hx * c + hy * s, hx * s + hy * c


def _extent(parts: list[_Part]) -> tuple[float, float]:
  """Half extents (x, y) of the obstacle's axis-aligned footprint."""
  half_x = max(abs(p.dx) + _part_extent(p)[0] for p in parts)
  half_y = max(_part_extent(p)[1] for p in parts)
  return half_x, half_y


def _overlaps(a, b, gap: float) -> bool:
  return not (
    a[1] + gap <= b[0] or b[1] + gap <= a[0] or a[3] + gap <= b[2] or b[3] + gap <= a[2]
  )


def _add_obstacle(
  body: mujoco.MjsBody,
  kind: str,
  parts: list[_Part],
  cx: float,
  cy: float,
  floor_at: FloorHeight,
) -> list[TerrainGeometry]:
  rgba = OBSTACLE_RGBA[kind]
  out = []
  for p in parts:
    px = cx + p.dx
    ex, ey = _part_extent(p)
    gx, gy = np.meshgrid(
      np.linspace(px - ex, px + ex, 7), np.linspace(cy - ey, cy + ey, 5)
    )
    under = floor_at(gx.ravel(), gy.ravel())
    z_lo = float(under.min()) - _EMBED
    z_top = float(under.max()) + p.height
    half_z = (z_top - z_lo) / 2
    pos = (px, cy, z_lo + half_z)
    if p.shape == "box":
      geom = body.add_geom(
        name=_name(kind),
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(p.half[0], p.half[1], half_z),
        pos=pos,
      )
      geom.quat = np.array([math.cos(p.yaw / 2), 0.0, 0.0, math.sin(p.yaw / 2)])
    else:
      geom = body.add_geom(
        name=_name(kind),
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=(p.half[0], half_z, 0.0),
        pos=pos,
      )
    geom.rgba[:] = rgba
    out.append(TerrainGeometry(geom=geom, color=rgba))
  return out


def _floor_boxes(
  body: mujoco.MjsBody, pieces: list[FloorPiece], width: float
) -> list[TerrainGeometry]:
  """Full-width boxes whose top faces are the profile's pieces.

  A flat piece reaches ``_SEAM_OVERLAP`` under a neighbour that is at least
  as high, so rays and feet never fall through an exact seam. Ramps and
  pieces next to lower neighbours stop at the edge: reaching further would
  leave a lip. The wedges between a ramp's slanted end faces and its
  neighbours lie below the surface, out of reach.
  """
  out = []
  for i, p in enumerate(pieces):
    lo = hi = 0.0
    if p.z0 == p.z1:
      if i > 0 and min(pieces[i - 1].z0, pieces[i - 1].z1) >= p.z0:
        lo = _SEAM_OVERLAP
      if i + 1 < len(pieces) and min(pieces[i + 1].z0, pieces[i + 1].z1) >= p.z1:
        hi = _SEAM_OVERLAP
    out.append(_floor_box(body, replace(p, x0=p.x0 - lo, x1=p.x1 + hi), width))
  return out


def _floor_box(body: mujoco.MjsBody, p: FloorPiece, width: float) -> TerrainGeometry:
  run, rise = p.x1 - p.x0, p.z1 - p.z0
  pitch = math.atan2(rise, run)
  # Top-face midpoint, then step down half a depth along the face normal.
  mid = np.array([(p.x0 + p.x1) / 2, width / 2, (p.z0 + p.z1) / 2])
  normal = np.array([-math.sin(pitch), 0.0, math.cos(pitch)])
  geom = body.add_geom(
    name=_name("floor"),
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(math.hypot(run, rise) / 2, width / 2, _FLOOR_DEPTH / 2),
    pos=(mid - normal * _FLOOR_DEPTH / 2).tolist(),
  )
  # Rotation about +y by -pitch tilts +x up by pitch.
  geom.quat = np.array([math.cos(-pitch / 2), 0.0, math.sin(-pitch / 2), 0.0])
  geom.rgba[:] = FLOOR_RGBA
  return TerrainGeometry(geom=geom, color=FLOOR_RGBA)
