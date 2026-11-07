"""Fault-free efficiency metrics for the household layouts."""

from __future__ import annotations

import numpy as np
from failbench.house import Item
from failbench.scripts.house_clean_report import robot_metrics, slowdown

ITEMS = (
  Item("floor", 0.0, 10.0, 0.0, 4.0, -1.0, 0.0),
  Item("person", 4.0, 4.4, 1.5, 1.8, 0.0, 1.7, "p"),
  Item("glass", 7.0, 7.1, 0.0, 0.8, 0.0, 1.0, "g"),
)
PATH = np.array([[0.0, 1.0], [10.0, 1.0]])


def test_tracking_error_and_clearances_are_measured_until_the_goal():
  # 0.1 m off the path for 2 s at 50 Hz, then 0.3 m off after reaching it.
  pos = np.zeros((150, 2))
  pos[:, 0] = np.linspace(0, 9, 150)
  pos[:100, 1] = 1.1
  pos[100:, 1] = 1.3
  m = robot_metrics(pos, PATH, ITEMS, time_to_goal_s=2.0, dt=0.02)
  assert abs(m["track_m"] - 0.1) < 1e-6
  # Before 2 s the robot reaches x = 6 (from x = 0 at 0.06/step): it passes
  # under the person 0.4 m off and never gets near the glass.
  assert abs(m["person_m"] - 0.4) < 1e-6
  assert m["glass_m"] > 1.0


def test_slowdown_pairs_robots_that_reached_the_goal_in_both_arms():
  base = {
    ("a", 1, 0): {"reached": True, "time": 10.0},
    ("a", 1, 1): {"reached": True, "time": 8.0},
    ("a", 1, 2): {"reached": False, "time": None},
  }
  arm = {
    ("a", 1, 0): {"reached": True, "time": 12.0},
    ("a", 1, 1): {"reached": False, "time": None},
    ("a", 1, 2): {"reached": True, "time": 9.0},
  }
  ratios = slowdown(base, arm, list(arm))
  assert ratios.tolist() == [1.2]
