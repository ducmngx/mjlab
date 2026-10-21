"""Normalisation statistics over the train split of built FailBench shards.

Writes ``<root>/stats.json``: per-channel mean and std of the vector inputs
and targets (``dv`` and ``pose`` only over upright, valid bins) and of the
clean map heights, plus split, family and fall counts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from failbench.data.build import SPLITS
from failbench.faults import FaultFamily

_VECTORS = ("state", "state_true_vel", "him_latent", "cmd_seq", "dv", "pose")


@dataclass(frozen=True)
class StatsConfig:
  root: str


class _Moments:
  def __init__(self):
    self.n, self.s, self.ss = 0, 0.0, 0.0

  def add(self, x: np.ndarray) -> None:
    x = x.reshape(-1, x.shape[-1]).astype(np.float64)
    self.n += len(x)
    self.s = self.s + x.sum(0)
    self.ss = self.ss + (x * x).sum(0)

  def result(self) -> dict[str, list[float]]:
    mean = self.s / max(self.n, 1)
    std = np.sqrt(np.maximum(self.ss / max(self.n, 1) - mean**2, 1e-12))
    return {"mean": np.atleast_1d(mean).tolist(), "std": np.atleast_1d(std).tolist()}


def main():
  cfg = tyro.cli(StatsConfig)
  root = Path(cfg.root)
  moments = {k: _Moments() for k in (*_VECTORS, "map")}
  counts = {
    "split": np.zeros(len(SPLITS), int),
    "family": np.zeros(len(FaultFamily), int),
  }
  fell = total = 0
  for meta in sorted(root.glob("*/meta.json")):
    d = meta.parent
    split = np.load(d / "split.npy")
    counts["split"] += np.bincount(split, minlength=len(SPLITS))
    train = split == SPLITS.index("train")
    counts["family"] += np.bincount(
      np.load(d / "fault_family.npy")[train], minlength=len(FaultFamily)
    )
    if not train.any():
      continue
    up = np.load(d / "upright.npy")[train] & np.load(d / "valid.npy")[train]
    fell += int((~up.all(-1)).sum())
    total += int(train.sum())
    for k in _VECTORS:
      x = np.load(d / f"{k}.npy", mmap_mode="r")[train]
      moments[k].add(x[up] if k in ("dv", "pose") else x)
    m = np.load(d / "map_clean.npy", mmap_mode="r")[train].astype(np.float32)
    moments["map"].add(m[~np.isnan(m)][:, None])
  out: dict[str, object] = {k: v.result() for k, v in moments.items()}
  out["counts"] = {
    "split": dict(zip(SPLITS, counts["split"].tolist(), strict=True)),
    "train_family": {
      f.name: int(c) for f, c in zip(FaultFamily, counts["family"], strict=True)
    },
    "train_fell_in_window": fell,
    "train_samples": total,
  }
  path = root / "stats.json"
  path.write_text(json.dumps(out, indent=2))
  print(f"[INFO] {json.dumps(out['counts'])} -> {path}")


if __name__ == "__main__":
  main()
