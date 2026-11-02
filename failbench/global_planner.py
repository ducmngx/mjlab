"""A NavFn-style global planner on a map's Nav2 costmap.

Nav2's default global planner (NavFn) finds the cheapest route on the
costmap by Dijkstra's search (its ``use_astar`` defaults to false), each
cell costing ``COST_NEUTRAL`` plus ``COST_FACTOR`` times its costmap value,
cells the robot's body cannot be in ruled out. Here the search runs on the
cell grid with 8 neighbours (``scipy``'s Dijkstra). A grid search's path
steps cell to cell, in 45 degree dog-legs where many routes cost the same
(NavFn descends a smooth potential instead), so the path is then pulled
straight wherever that brings it no closer to an obstacle than it was
(``_shortcut``) and its corners rounded (``_smooth``). Paths are planned
once, offline, and reused (``failbench/scripts/plan_paths.py``).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from failbench.mppi.nav2 import INSCRIBED

COST_NEUTRAL = 50.0
COST_FACTOR = 0.8
"""NavFn's cell cost: ``COST_NEUTRAL + COST_FACTOR * costmap``."""


def _graph(cost: np.ndarray) -> coo_matrix:
  """The grid's 8-neighbour graph: an edge between free cells weighs their
  mean cell cost times the step's length (in cells)."""
  h, w = cost.shape
  free = cost < INSCRIBED
  cell = COST_NEUTRAL + COST_FACTOR * np.minimum(cost, INSCRIBED - 1)
  idx = np.arange(h * w).reshape(h, w)
  rows, cols, weights = [], [], []
  for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
    ys = slice(0, h - dy)
    xa = slice(max(0, -dx), w - max(0, dx))
    xb = slice(max(0, dx), w + min(0, dx))
    a, b = idx[ys, xa], idx[dy:, xb]
    ok = free[ys, xa] & free[dy:, xb]
    weight = (cell[ys, xa] + cell[dy:, xb]) / 2 * np.hypot(dx, dy)
    rows += [a[ok], b[ok]]
    cols += [b[ok], a[ok]]
    weights += [weight[ok], weight[ok]]
  n = h * w
  return coo_matrix(
    (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
    shape=(n, n),
  )


def _shortcut(xy: np.ndarray, cost: np.ndarray, resolution: float) -> np.ndarray:
  """The path ``[L, 2]`` pulled straight: from each kept point, on to the
  furthest point the straight segment to which crosses no cell costlier
  than the path's own costliest cell between them."""
  h, w = cost.shape

  def cells(pts: np.ndarray) -> np.ndarray:
    col = np.round(pts[..., 0] / resolution + (w - 1) / 2).astype(int).clip(0, w - 1)
    row = np.round(pts[..., 1] / resolution + (h - 1) / 2).astype(int).clip(0, h - 1)
    return cost[row, col]

  own = cells(xy)
  worst = np.maximum.accumulate  # costliest so far, along the path
  keep, i, n = [0], 0, len(xy)
  t = np.linspace(0, 1, 64)[:, None]
  while i < n - 1:
    ahead = np.arange(n - 1, i, -1)
    # The path's costliest cell from i to each j, and each segment's.
    limit = worst(own[i:])[ahead - i]
    seg = xy[i] + t[None] * (xy[ahead][:, None] - xy[i])  # [J, 64, 2]
    ok = cells(seg).max(-1) <= limit
    j = int(ahead[np.argmax(ok)]) if ok.any() else i + 1
    keep.append(j)
    i = j
  return xy[keep]


def _densify(xy: np.ndarray, resolution: float) -> np.ndarray:
  """Points every ``resolution`` metres or less along a polyline."""
  out = [xy[:1]]
  for a, b in zip(xy[:-1], xy[1:], strict=True):
    k = max(1, int(np.ceil(np.linalg.norm(b - a) / resolution)))
    out.append(a + np.linspace(0, 1, k + 1)[1:, None] * (b - a))
  return np.concatenate(out)


def _smooth(path: np.ndarray, window: int) -> np.ndarray:
  """A moving average over ``window`` points, the ends kept."""
  if len(path) <= 2 or window < 2:
    return path
  k = min(window, len(path))
  pad = np.concatenate(
    [np.repeat(path[:1], k // 2, 0), path, np.repeat(path[-1:], k // 2, 0)]
  )
  kernel = np.ones(k) / k
  out = np.stack([np.convolve(pad[:, i], kernel, mode="valid") for i in range(2)], -1)
  out = out[: len(path)]
  out[0], out[-1] = path[0], path[-1]
  return out


def plan_paths(
  cost: np.ndarray,
  resolution: float,
  starts: np.ndarray,
  goals: np.ndarray,
  smooth_m: float = 0.5,
) -> list[np.ndarray | None]:
  """Paths ``[L, 2]`` (map frame, metres; the grid centred on it) on one
  costmap ``[H, W]`` from each of ``starts [K, 2]`` to its goal in
  ``goals [K, 2]``; None where there is none (an end in or walled off by
  obstacles)."""
  h, w = cost.shape

  def cell(xy: np.ndarray) -> np.ndarray:
    col = np.round(xy[:, 0] / resolution + (w - 1) / 2).astype(int)
    row = np.round(xy[:, 1] / resolution + (h - 1) / 2).astype(int)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    return np.where(inside, row.clip(0, h - 1) * w + col.clip(0, w - 1), -1)

  src, dst = cell(starts), cell(goals)
  flat = cost.reshape(-1)
  usable = (src >= 0) & (dst >= 0)
  usable &= (flat[src.clip(0)] < INSCRIBED) & (flat[dst.clip(0)] < INSCRIBED)
  out: list[np.ndarray | None] = [None] * len(starts)
  if not usable.any():
    return out
  graph = _graph(cost).tocsr()
  unique, inverse = np.unique(src[usable], return_inverse=True)
  dist, pred = dijkstra(graph, indices=unique, return_predecessors=True)
  window = max(1, round(smooth_m / resolution))
  for k, row in zip(np.nonzero(usable)[0], inverse, strict=True):
    if not np.isfinite(dist[row, dst[k]]):
      continue
    node, chain = dst[k], [dst[k]]
    while node != src[k]:
      node = pred[row, node]
      chain.append(node)
    chain = np.array(chain[::-1])
    xy = np.stack(
      [(chain % w - (w - 1) / 2) * resolution, (chain // w - (h - 1) / 2) * resolution],
      -1,
    )
    xy[0], xy[-1] = starts[k], goals[k]
    xy = _densify(_shortcut(xy, cost, resolution), resolution)
    out[k] = _smooth(xy, window)
  return out
