"""Phantom maps: the layouts the planner-driven data's global paths are
planned on.

A map is a square of ``size_m`` metres with up to ``max_boxes`` boxes (the
phantom format: centre x, y, half x, half y, yaw; in the map's frame,
centred on it), drawn from one of the ``LAYOUTS``. Like a trial's phantoms
(``failbench/phantoms.py``) a map exists only in the planners' costmaps,
never in the physics, so the drift the impact model learns stays
obstacle-free; the map shapes the route (``failbench/global_planner.py``)
and the local planner's moves around it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from failbench.mppi.nav2 import CostmapCfg, inflate

LAYOUTS = ("clutter", "rooms", "corridors", "open")
"""A scatter of boxes; a grid of rooms joined by doors; long walls with a
gap each, so the way through zigzags; or a few boxes on open ground."""


@dataclass(frozen=True)
class MapCfg:
  size_m: float = 12.0
  resolution: float = 0.05
  """The occupancy grid's cell: the local planners' costmap resolution."""
  max_boxes: int = 40
  costmap: CostmapCfg = field(default_factory=CostmapCfg)
  """The inflation the global planner plans on (the local planners')."""
  clutter_count: tuple[int, int] = (12, 28)
  clutter_size_m: tuple[float, float] = (0.3, 1.5)
  rooms: tuple[int, int] = (2, 3)
  """Rooms along each side (uniform, inclusive)."""
  door_m: tuple[float, float] = (1.0, 1.6)
  wall_m: float = 0.15
  """Wall thickness."""
  furniture: tuple[int, int] = (0, 5)
  """Small boxes scattered in a rooms map."""
  corridor_walls: tuple[int, int] = (2, 4)
  gap_m: tuple[float, float] = (1.0, 1.8)
  open_count: tuple[int, int] = (0, 3)

  @property
  def cells(self) -> int:
    return round(self.size_m / self.resolution)


def _box(x: float, y: float, hx: float, hy: float, yaw: float = 0.0) -> list[float]:
  return [x, y, hx, hy, yaw]


def _scatter(
  rng: np.random.Generator, count: int, size: tuple[float, float], half: float
) -> list:
  out = []
  for _ in range(count):
    hx, hy = rng.uniform(size[0], size[1], 2) / 2
    x, y = rng.uniform(-half + 0.5, half - 0.5, 2)
    out.append(_box(x, y, hx, hy, rng.uniform(-np.pi, np.pi)))
  return out


def _wall_with_door(
  rng: np.random.Generator,
  along_x: bool,
  at: float,
  lo: float,
  hi: float,
  door: float,
  thick: float,
) -> list:
  """A wall at ``at`` (y if ``along_x`` else x) from ``lo`` to ``hi`` with
  a ``door``-wide gap at a random place."""
  start = rng.uniform(lo + 0.3, max(lo + 0.3, hi - 0.3 - door))
  out = []
  for a, b in ((lo, start), (start + door, hi)):
    if b - a > 0.05:
      mid, half = (a + b) / 2, (b - a) / 2
      out.append(
        _box(mid, at, half, thick / 2) if along_x else _box(at, mid, thick / 2, half)
      )
  return out


def sample_map(kind: str, rng: np.random.Generator, cfg: MapCfg) -> np.ndarray:
  """One map of ``kind`` (``LAYOUTS``): boxes ``[max_boxes, 5]``, unused
  rows with half sizes 0."""
  half = cfg.size_m / 2
  boxes: list = []
  if kind == "clutter":
    count = int(rng.integers(cfg.clutter_count[0], cfg.clutter_count[1] + 1))
    boxes = _scatter(rng, count, cfg.clutter_size_m, half)
  elif kind == "rooms":
    nx, ny = (int(k) for k in rng.integers(cfg.rooms[0], cfg.rooms[1] + 1, 2))
    xs, ys = np.linspace(-half, half, nx + 1), np.linspace(-half, half, ny + 1)
    for x in xs[1:-1]:  # Walls between columns of rooms, a door per room.
      for y0, y1 in zip(ys[:-1], ys[1:], strict=True):
        door = rng.uniform(*cfg.door_m)
        boxes += _wall_with_door(rng, False, x, y0, y1, door, cfg.wall_m)
    for y in ys[1:-1]:
      for x0, x1 in zip(xs[:-1], xs[1:], strict=True):
        door = rng.uniform(*cfg.door_m)
        boxes += _wall_with_door(rng, True, y, x0, x1, door, cfg.wall_m)
    count = int(rng.integers(cfg.furniture[0], cfg.furniture[1] + 1))
    boxes += _scatter(rng, count, (0.3, 0.8), half)
  elif kind == "corridors":
    count = int(rng.integers(cfg.corridor_walls[0], cfg.corridor_walls[1] + 1))
    along_x = bool(rng.integers(2))
    for at in np.linspace(-half, half, count + 2)[1:-1]:
      gap = rng.uniform(*cfg.gap_m)
      boxes += _wall_with_door(rng, along_x, at, -half, half, gap, cfg.wall_m)
  elif kind == "open":
    count = int(rng.integers(cfg.open_count[0], cfg.open_count[1] + 1))
    boxes = _scatter(rng, count, cfg.clutter_size_m, half)
  else:
    raise ValueError(f"Unknown layout {kind!r}; expected {LAYOUTS}.")
  out = np.zeros((cfg.max_boxes, 5), np.float32)
  boxes = boxes[: cfg.max_boxes]
  if boxes:
    out[: len(boxes)] = np.array(boxes, np.float32)
  return out


def rasterize(
  boxes: torch.Tensor, cells: int, resolution: float, chunk: int = 64
) -> torch.Tensor:
  """Occupancy grids ``[M, cells, cells]`` (row = y, column = x, centred
  on the map) of maps' boxes ``[M, B, 5]``, about ``chunk`` map-boxes at a
  time."""
  r = (torch.arange(cells, device=boxes.device) - (cells - 1) / 2) * resolution
  y, x = torch.meshgrid(r, r, indexing="ij")
  out = torch.zeros(len(boxes), cells, cells, dtype=torch.bool, device=boxes.device)
  per = max(1, min(boxes.shape[1], chunk))
  maps = max(1, chunk // per)
  for i in range(0, len(boxes), maps):
    out[i : i + maps] = _rasterize(boxes[i : i + maps], x, y, per)
  return out


def _rasterize(boxes: torch.Tensor, x: torch.Tensor, y: torch.Tensor, per: int):
  out = torch.zeros(len(boxes), *x.shape, dtype=torch.bool, device=boxes.device)
  for b in boxes.split(per, dim=1):
    rel_x = x[None, None] - b[..., 0, None, None]  # [M, chunk, H, W]
    rel_y = y[None, None] - b[..., 1, None, None]
    c, s = torch.cos(b[..., 4])[..., None, None], torch.sin(b[..., 4])[..., None, None]
    lx, ly = c * rel_x + s * rel_y, -s * rel_x + c * rel_y
    inside = (lx.abs() <= b[..., 2, None, None]) & (ly.abs() <= b[..., 3, None, None])
    out |= (inside & (b[..., 2, None, None] > 0)).any(1)
  return out


def map_costmap(boxes: torch.Tensor, cfg: MapCfg) -> torch.Tensor:
  """The Nav2 costmaps ``[M, cells, cells]`` of maps' boxes ``[M, B, 5]``."""
  occ = rasterize(boxes, cfg.cells, cfg.resolution)
  return inflate(occ, cfg.resolution, cfg.costmap)
