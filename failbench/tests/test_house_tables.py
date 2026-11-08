"""The household experiment's paper tables."""

from __future__ import annotations

from failbench.house import LAYOUTS
from failbench.scripts.house_tables import FORCES, SAFETY, tables


def _fault(mean, diff=None):
  stats: dict = {"n": 4}
  for m, *_ in SAFETY + FORCES:
    stats[m] = {"mean": mean} | ({"diff": diff, "se": 0.01} if diff is not None else {})
  return {"all": stats} | {lay: stats for lay in LAYOUTS}


def _clean(slow):
  s = {
    "n": 4,
    "reached": 1.0,
    "slow_mean": slow,
    "slow_median": slow,
    "slow_p90": slow,
    "time_s": 10.0,
    "track_cm": 5.0,
    "close_person": 0.1,
    "close_glass": 0.0,
    "touched": 0.0,
    "fell": 0.0,
    "plan_ms": 100.0,
  }
  return {"all": s} | {lay: s for lay in LAYOUTS}


def test_tables_list_normal_then_ours_then_baselines_with_paired_changes():
  f = {
    "inflate_0.1": _fault(0.1, 0.0),
    "normal": _fault(0.1),
    "ours_w1": _fault(0.05, -0.05),
  }
  e = {"inflate_0.1": _clean(1.02), "normal": _clean(1.0), "ours_w1": _clean(1.1)}
  md = tables(f, e)
  rows = [ln.split(" | ")[0] for ln in md.splitlines() if ln.startswith("| ")]
  first = rows[1:4]
  assert first == ["| normal", "| ours_w1", "| inflate_0.1"]
  assert "5.0% (-5.0 ± 1.0)" in md  # ours' collision rate and its paired change.
  assert "| ours_w1 | +10% | 100.0% | -50% | -50% | -50% |" in md
