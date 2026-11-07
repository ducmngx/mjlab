"""Fault-free efficiency on the household layouts (MODE=clean runs of
failbench/cluster/house_faults.sbatch).

Reads ``<root>/<arm>/<layout>_s<seed>.{json,npz}`` and reports per arm, pooled
and per layout: success and fall rates, how often the robot touched an
object, the time to goal against ``normal`` (paired by layout, seed and robot,
over robots that reached the goal in both), how far it strayed from the
global path, and how close it passed people and glass on the way. Writes
``<out>.json`` and prints a table.

  uv run python -m failbench.scripts.house_clean_report --root house_clean --out eff
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from failbench.house import LAYOUTS, Item
from failbench.house_terrain import route_table

DT = 0.02
CLOSE_PERSON_M = 0.5
CLOSE_GLASS_M = 0.3


@dataclass
class CleanReportConfig:
  root: str
  out: str
  baseline: str = "normal"


def _path_dist(p: np.ndarray, path: np.ndarray) -> np.ndarray:
  """Distance from each point ``[T, 2]`` to the polyline ``path [L, 2]``."""
  a, b = path[:-1], path[1:]
  ab = b - a
  t = ((p[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12)
  q = a[None] + t.clip(0, 1)[..., None] * ab[None]
  return np.linalg.norm(p[:, None] - q, axis=-1).min(1)


def _rect_dist(p: np.ndarray, it: Item) -> np.ndarray:
  dx = np.maximum(np.maximum(it.x0 - p[:, 0], 0), p[:, 0] - it.x1)
  dy = np.maximum(np.maximum(it.y0 - p[:, 1], 0), p[:, 1] - it.y1)
  return np.hypot(dx, dy)


def robot_metrics(
  pos: np.ndarray,
  path: np.ndarray,
  items: Sequence[Item],
  time_to_goal_s: float | None,
  dt: float = DT,
) -> dict:
  """One robot's trunk track ``pos [T, 2]`` (layout frame), up to the goal:
  mean distance from its global ``path``, and the closest it came to a
  person and to glass (inf: none in the layout)."""
  end = len(pos) if time_to_goal_s is None else max(1, round(time_to_goal_s / dt))
  p = pos[:end]
  out = {"track_m": float(_path_dist(p, path).mean())}
  for cls in ("person", "glass"):
    ds = [float(_rect_dist(p, it).min()) for it in items if it.cls == cls]
    out[f"{cls}_m"] = min(ds) if ds else float("inf")
  return out


def slowdown(base: dict, arm: dict, keys) -> np.ndarray:
  """Time to goal over the baseline's, for the robots in ``keys`` that
  reached the goal in both."""
  return np.array(
    [
      arm[k]["time"] / base[k]["time"]
      for k in keys
      if k in base and arm[k]["reached"] and base[k]["reached"]
    ]
  )


def _rows(root: Path) -> dict[str, dict[tuple, dict]]:
  out: dict[str, dict[tuple, dict]] = {}
  tables = {name: route_table(name) for name in LAYOUTS}
  for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    rows = {}
    for f in sorted(arm_dir.glob("*_s*.json")):
      layout, seed = f.stem.rsplit("_s", 1)
      j = json.load(open(f))
      pos = np.load(f.with_suffix(".npz"))["dense_base_pos_w"][..., :2]
      tab, paths = tables[layout]
      n = len(pos)
      org = pos[:, 0] - tab[np.arange(n) % len(tab), :2]
      for t in j["trials"]:
        b = t["env"]
        m = robot_metrics(
          pos[b] - org[b],
          paths[b % len(paths)],
          LAYOUTS[layout].items,
          t["time_to_goal_s"],
        )
        m |= {
          "reached": bool(t["reached"]),
          "time": t["time_to_goal_s"],
          "fell": bool(t["fell"]),
          "touched": float(t["incident_severity"]) > 0,
          "plan_ms": j["summary"].get("plan_ms_mean"),
        }
        rows[(layout, int(seed), b)] = m
    out[arm_dir.name] = rows
  return out


def _stats(rows: dict, base: dict, keys: list) -> dict:
  r = [rows[k] for k in keys]
  ratio = slowdown(base, rows, keys)
  done = [x["time"] for x in r if x["reached"]]
  ms = [x["plan_ms"] for x in r if x["plan_ms"] is not None]
  return {
    "n": len(r),
    "reached": float(np.mean([x["reached"] for x in r])),
    "fell": float(np.mean([x["fell"] for x in r])),
    "touched": float(np.mean([x["touched"] for x in r])),
    "time_s": float(np.mean(done)) if done else None,
    "slow_mean": float(ratio.mean()) if len(ratio) else None,
    "slow_median": float(np.median(ratio)) if len(ratio) else None,
    "slow_p90": float(np.percentile(ratio, 90)) if len(ratio) else None,
    "track_cm": 100 * float(np.mean([x["track_m"] for x in r])),
    "person_med_m": float(np.median([x["person_m"] for x in r])),
    "glass_med_m": float(np.median([x["glass_m"] for x in r])),
    "close_person": float(np.mean([x["person_m"] < CLOSE_PERSON_M for x in r])),
    "close_glass": float(np.mean([x["glass_m"] < CLOSE_GLASS_M for x in r])),
    "plan_ms": float(np.mean(ms)) if ms else None,
  }


COLUMNS = {
  "reached": ("success", "{:.3f}"),
  "fell": ("fell", "{:.3f}"),
  "touched": ("touched", "{:.3f}"),
  "time_s": ("time s", "{:.1f}"),
  "slow_mean": ("slow", "{:.2f}"),
  "slow_p90": ("slow p90", "{:.2f}"),
  "track_cm": ("track cm", "{:.1f}"),
  "close_person": (f"<{CLOSE_PERSON_M}m pers", "{:.3f}"),
  "close_glass": (f"<{CLOSE_GLASS_M}m glass", "{:.3f}"),
  "plan_ms": ("plan ms", "{:.0f}"),
}


def _fmt(v, f: str) -> str:
  return "-" if v is None else f.format(v)


def main(cfg: CleanReportConfig) -> None:
  data = _rows(Path(cfg.root))
  if cfg.baseline not in data:
    raise ValueError(f"No {cfg.baseline!r} arm under {cfg.root}.")
  base = data[cfg.baseline]
  report: dict = {}
  for arm, rows in data.items():
    keys = list(rows)
    report[arm] = {"all": _stats(rows, base, keys)}
    for lay in LAYOUTS:
      lk = [k for k in keys if k[0] == lay]
      if lk:
        report[arm][lay] = _stats(rows, base, lk)
  json.dump(report, open(Path(cfg.out).with_suffix(".json"), "w"), indent=1)
  print(f"{'arm':14s}" + "".join(f"{lab:>14s}" for lab, _ in COLUMNS.values()))
  for arm in data:
    s = report[arm]["all"]
    print(
      f"{arm:14s}" + "".join(f"{_fmt(s[k], f):>14s}" for k, (_, f) in COLUMNS.items())
    )
  print("\nslowdown per layout (mean, robots that reached in both)")
  print(f"{'arm':14s}" + "".join(f"{lay:>11s}" for lay in LAYOUTS))
  for arm in data:
    cells = [
      _fmt(report[arm].get(lay, {}).get("slow_mean"), "{:.2f}") for lay in LAYOUTS
    ]
    print(f"{arm:14s}" + "".join(f"{c:>11s}" for c in cells))
  print("\nsuccess per layout")
  for arm in data:
    cells = [_fmt(report[arm].get(lay, {}).get("reached"), "{:.2f}") for lay in LAYOUTS]
    print(f"{arm:14s}" + "".join(f"{c:>11s}" for c in cells))


if __name__ == "__main__":
  main(tyro.cli(CleanReportConfig))
