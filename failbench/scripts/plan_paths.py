"""Plan a library of global paths on phantom maps, for the planner-driven data.

Draws ``maps`` maps (``failbench/maps.py``, the ``LAYOUTS`` in turn), and on
each plans ``pairs`` paths between random free start and goal points at
least ``min_length_m`` apart (``failbench/global_planner.py``). Writes one
``.npz``:

- ``boxes [M, B, 5]`` each map's boxes (map frame), ``layout [M]`` its
  layout (an index into ``LAYOUTS``), ``size_m``, ``resolution``;
- ``paths [K, P, 2]`` each path's points every ``resolution`` metres (map
  frame), ``real [K]`` how many are its own (the rest repeat the goal),
  ``path_map [K]`` its map, ``length_m [K]``.

  uv run python -m failbench.scripts.plan_paths --output paths.npz
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

from failbench.global_planner import plan_paths
from failbench.maps import LAYOUTS, MapCfg, map_costmap, sample_map
from failbench.paths import resample


@dataclass
class PlanPathsConfig:
  output: str
  maps: int = 200
  pairs: int = 32
  """Start-goal pairs tried per map."""
  min_length_m: float = 4.0
  """Straight-line distance between a start and its goal, at least."""
  max_points: int = 600
  """Points kept per path (30 m at 0.05 m); longer paths are dropped."""
  clear_cost: float = 50.0
  """Starts and goals only where the costmap is at most this (clear of the
  inflation's steep part)."""
  seed: int = 0
  map: MapCfg = field(default_factory=MapCfg)


def main(cfg: PlanPathsConfig) -> None:
  rng = np.random.default_rng(cfg.seed)
  mc = cfg.map
  boxes = np.stack(
    [sample_map(LAYOUTS[i % len(LAYOUTS)], rng, mc) for i in range(cfg.maps)]
  )
  costs = map_costmap(torch.from_numpy(boxes), mc).numpy()
  r = (np.arange(mc.cells) - (mc.cells - 1) / 2) * mc.resolution
  ys, xs = np.meshgrid(r, r, indexing="ij")
  paths, reals, path_map = [], [], []
  t0 = time.time()
  for m in range(cfg.maps):
    clear = costs[m] <= cfg.clear_cost
    pts = np.stack([xs[clear], ys[clear]], -1)
    if len(pts) < 2:
      continue
    starts = pts[rng.integers(len(pts), size=cfg.pairs * 4)]
    goals = pts[rng.integers(len(pts), size=cfg.pairs * 4)]
    far = np.linalg.norm(goals - starts, axis=-1) >= cfg.min_length_m
    starts, goals = starts[far][: cfg.pairs], goals[far][: cfg.pairs]
    for path in plan_paths(costs[m], mc.resolution, starts, goals):
      if path is None:
        continue
      res, real = resample(
        torch.from_numpy(path[None]).float(), mc.resolution, cfg.max_points + 1
      )
      if int(real[0]) > cfg.max_points:
        continue
      paths.append(res[0, : cfg.max_points].numpy())
      reals.append(int(real[0]))
      path_map.append(m)
    if (m + 1) % 50 == 0:
      print(
        f"[INFO] {m + 1}/{cfg.maps} maps, {len(paths)} paths ({time.time() - t0:.0f} s)"
      )
  real = np.array(reals)
  out = Path(cfg.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out,
    boxes=boxes,
    layout=np.array([i % len(LAYOUTS) for i in range(cfg.maps)], np.uint8),
    size_m=np.array(mc.size_m),
    resolution=np.array(mc.resolution),
    paths=np.stack(paths).astype(np.float32),
    real=real,
    path_map=np.array(path_map, np.int32),
    length_m=(real - 1) * mc.resolution,
  )
  print(f"[INFO] {len(paths)} paths on {cfg.maps} maps -> {out}")


if __name__ == "__main__":
  main(tyro.cli(PlanPathsConfig))
