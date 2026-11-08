"""Markdown result tables for the household experiment, for the paper.

Reads the two reports (``house_report``'s with faults, ``house_clean_report``'s
fault-free) and prints the safety, distance/force, efficiency, summary and
per-layout tables (failbench/experiments/house_faults_results.md). Arms in
both reports are listed: ``normal``, then ours, then the baselines.

  uv run python -m failbench.scripts.house_tables --faults report.json --clean eff.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import tyro

from failbench.house import LAYOUTS


@dataclass
class TablesConfig:
  faults: str
  clean: str
  baseline: str = "normal"


def _cell(s: dict, m: str, pct: bool) -> str:
  x = s[m]
  v = x["mean"]
  if v is None:
    return "-"
  t = f"{100 * v:.1f}%" if pct else f"{v:.2f}"
  if "diff" in x:
    d, se = x["diff"], x["se"]
    t += f" ({100 * d:+.1f} ± {100 * se:.1f})" if pct else f" ({d:+.2f} ± {se:.2f})"
  return t


def _row(name: str, cells: list[str]) -> str:
  return f"| {name} | " + " | ".join(cells) + " |"


def _head(cols: list[str]) -> str:
  return "| Arm | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols)


SAFETY = [
  ("collided", "Collision", True),
  ("collided_person", "Person hit", True),
  ("collided_glass", "Glass hit", True),
  ("hard_collided_person", "Hard person hit", True),
  ("damage", "Damage", False),
  ("close_person", "Within 0.2 m of a person", True),
  ("near_miss", "Person near miss", True),
  ("fell", "Fell", True),
]
FORCES = [
  ("min_dist_person_m", "Closest to a person (m)"),
  ("end_dist_person_m", "Nearest person at 3 s (m)"),
  ("end_dist_m", "Nearest object at 3 s (m)"),
  ("peak_force_n", "Peak force on an object (N)"),
  ("peak_force_person_n", "Peak force on a person (N)"),
  ("impulse_ns", "Impulse (N s)"),
  ("impact_speed_hit", "Impact speed, robots that touched (m/s)"),
]


def tables(f: dict, e: dict, baseline: str = "normal") -> str:
  others = (a for a in f if a in e and a != baseline)
  arms = [baseline] + sorted(others, key=lambda a: (not a.startswith("ours"), a))
  lays = list(LAYOUTS)
  out = ["### Safety after a fault (3 s window), all layouts\n"]
  out.append(_head([c[1] for c in SAFETY] + ["n"]))
  for a in arms:
    s = f[a]["all"]
    out.append(_row(a, [_cell(s, m, p) for m, _, p in SAFETY] + [str(s["n"])]))
  out.append(
    f"\nPaired difference to `{baseline}` ± standard error in brackets "
    "(percentage points for rates).\n"
  )
  out.append("### Distances and forces after a fault\n")
  out.append(_head([c[1] for c in FORCES]))
  for a in arms:
    out.append(_row(a, [_cell(f[a]["all"], m, False) for m, _ in FORCES]))
  out.append("\n### Fault-free efficiency, all layouts\n")
  out.append(
    _head(
      [
        "Success",
        "Slowdown vs normal (mean)",
        "Slowdown (median)",
        "Slowdown (p90)",
        "Time to goal (s)",
        "Path error (cm)",
        "Within 0.5 m of a person",
        "Within 0.3 m of glass",
        "Touched",
        "Fell",
        "Plan time per step, 64 robots (ms)",
      ]
    )
  )
  for a in arms:
    s = e[a]["all"]
    out.append(
      _row(
        a,
        [
          f"{100 * s['reached']:.1f}%",
          f"{s['slow_mean']:.2f}",
          f"{s['slow_median']:.2f}",
          f"{s['slow_p90']:.2f}",
          f"{s['time_s']:.1f}",
          f"{s['track_cm']:.1f}",
          f"{100 * s['close_person']:.1f}%",
          f"{100 * s['close_glass']:.1f}%",
          f"{100 * s['touched']:.1f}%",
          f"{100 * s['fell']:.1f}%",
          f"{s['plan_ms']:.0f}",
        ],
      )
    )
  out.append(f"\nn = {e[baseline]['all']['n']} robots per arm.\n")
  out.append(f"### Summary: change from `{baseline}`\n")
  out.append(
    _head(
      [
        "Fault-free slowdown",
        "Fault-free success",
        "Collisions",
        "Person hits",
        "Damage",
      ]
    )
  )
  n = f[baseline]["all"]
  for a in arms[1:]:
    s, c = f[a]["all"], e[a]["all"]

    def rel(m, s=s):
      return f"{100 * (s[m]['mean'] / n[m]['mean'] - 1):+.0f}%"

    out.append(
      _row(
        a,
        [
          f"{100 * (c['slow_mean'] - 1):+.0f}%",
          f"{100 * c['reached']:.1f}%",
          rel("collided"),
          rel("collided_person"),
          rel("damage"),
        ],
      )
    )
  for title, m, pct in (
    ("Collision rate after a fault, per layout", "collided", True),
    ("Damage after a fault, per layout", "damage", False),
  ):
    out.append(f"\n### {title}\n")
    out.append(_head(lays))
    for a in arms:
      out.append(_row(a, [_cell(f[a][lay], m, pct) for lay in lays]))
  out.append("\n### Fault-free slowdown and success, per layout\n")
  out.append(_head(lays))
  for a in arms:
    out.append(
      _row(
        a,
        [
          f"{e[a][lay]['slow_mean']:.2f} / {100 * e[a][lay]['reached']:.0f}%"
          for lay in lays
        ],
      )
    )
  out.append("\nEach cell: mean slowdown / success.")
  return "\n".join(out)


def main(cfg: TablesConfig) -> None:
  print(tables(json.load(open(cfg.faults)), json.load(open(cfg.clean)), cfg.baseline))


if __name__ == "__main__":
  main(tyro.cli(TablesConfig))
