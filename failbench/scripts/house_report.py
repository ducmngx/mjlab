"""Aggregate the household fault experiment (failbench/experiments/house_faults.md).

Reads ``<root>/<arm>/<layout>_s<seed>.json`` and reports, per arm, pooled and
per layout, the window metrics over robots whose window completed: rates and
means, and each arm's paired difference to ``normal`` (same layout, seed and
robot) with its standard error. Writes ``<out>.json`` and prints a table.

  uv run python -m failbench.scripts.house_report --root house_faults --out report
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

NEAR_MISS_M = 0.1

METRICS = {
  "collided": "collision rate",
  "collided_person": "person collision rate",
  "collided_glass": "glass collision rate",
  "end_dist_m": "distance to nearest object at 3 s (m)",
  "end_dist_person_m": "distance to nearest person at 3 s (m)",
  "min_dist_person_m": "closest to a person in the window (m)",
  "peak_force_n": "peak contact force on an object (N)",
  "peak_force_person_n": "peak contact force on a person (N)",
  "impulse_ns": "contact impulse on objects (N s)",
  "damage": "damage",
  "fell": "fall rate",
  "near_miss": f"person near miss (< {NEAR_MISS_M} m, no touch)",
}


@dataclass
class ReportConfig:
  root: str
  out: str
  baseline: str = "normal"


def _rows(root: Path) -> dict[str, dict[tuple, dict]]:
  """``{arm: {(layout, seed, robot): window metrics}}``."""
  out: dict[str, dict[tuple, dict]] = {}
  for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    rows = {}
    for f in sorted(arm_dir.glob("*_s*.json")):
      layout, seed = f.stem.rsplit("_s", 1)
      for t in json.load(open(f))["trials"]:
        w = t.get("window")
        if not w or not w["complete"]:
          continue
        w = dict(w)
        w["near_miss"] = (
          w["min_dist_person_m"] < NEAR_MISS_M and not w["collided_person"]
        )
        rows[(layout, int(seed), t["env"])] = w
    out[arm_dir.name] = rows
  return out


def _stats(rows: dict[tuple, dict], base: dict[tuple, dict] | None, sel) -> dict:
  keys = [k for k in rows if sel(k)]
  res: dict = {"n": len(keys)}
  for m in METRICS:
    v = np.array([float(rows[k].get(m, np.nan)) for k in keys])
    v = v[np.isfinite(v)]
    res[m] = {"mean": float(v.mean()) if len(v) else None}
    if base is not None:
      pk = [k for k in keys if k in base]
      a = np.array([float(base[k].get(m, np.nan)) for k in pk])
      b = np.array([float(rows[k].get(m, np.nan)) for k in pk])
      ok = np.isfinite(a) & np.isfinite(b)
      d = b[ok] - a[ok]
      if len(d) > 1:
        res[m]["diff"] = float(d.mean())
        res[m]["se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        res[m]["paired_n"] = int(len(d))
  return res


def main(cfg: ReportConfig) -> None:
  data = _rows(Path(cfg.root))
  if cfg.baseline not in data:
    raise ValueError(f"No {cfg.baseline!r} arm under {cfg.root}.")
  base = data[cfg.baseline]
  layouts = sorted({k[0] for rows in data.values() for k in rows})
  report: dict = {}
  for arm, rows in data.items():
    b = None if arm == cfg.baseline else base
    report[arm] = {"all": _stats(rows, b, lambda k: True)}
    for lay in layouts:
      report[arm][lay] = _stats(rows, b, lambda k, lay=lay: k[0] == lay)
  json.dump(report, open(Path(cfg.out).with_suffix(".json"), "w"), indent=1)
  arms = list(data)
  print(f"{'metric':44s}" + "".join(f"{a:>22s}" for a in arms))
  for m, label in METRICS.items():
    cells = []
    for a in arms:
      s = report[a]["all"][m]
      if s["mean"] is None:
        cells.append(f"{'-':>22s}")
        continue
      txt = f"{s['mean']:.3f}"
      if "diff" in s:
        txt += f" ({s['diff']:+.3f}±{s['se']:.3f})"
      cells.append(f"{txt:>22s}")
    print(f"{label:44s}" + "".join(cells))
  print("robots:", {a: report[a]["all"]["n"] for a in arms})


if __name__ == "__main__":
  main(tyro.cli(ReportConfig))
