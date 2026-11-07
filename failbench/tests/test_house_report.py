"""The household report's metrics derived from each robot's fault window."""

from __future__ import annotations

import math

from failbench.scripts.house_report import derived


def _window(**kw):
  w = {
    "collided": False,
    "collided_person": False,
    "impact_speed": float("nan"),
    "min_dist_person_m": 1.0,
  }
  w.update(kw)
  return w


def test_a_slow_graze_is_a_collision_but_not_a_hard_one():
  d = derived(
    _window(
      collided=True, collided_person=True, impact_speed=0.07, min_dist_person_m=0.0
    )
  )
  assert not d["hard_collided"] and not d["hard_collided_person"]
  assert d["impact_speed_hit"] == 0.07
  assert d["near_miss"] is False and d["close_person"] is True


def test_a_fast_hit_on_a_person_is_a_hard_person_collision():
  d = derived(
    _window(
      collided=True, collided_person=True, impact_speed=0.4, min_dist_person_m=0.0
    )
  )
  assert d["hard_collided"] and d["hard_collided_person"]


def test_a_close_pass_without_touching_is_a_near_miss_and_has_no_impact_speed():
  d = derived(_window(min_dist_person_m=0.06))
  assert d["near_miss"] is True and d["close_person"] is True
  assert not d["hard_collided"]
  assert math.isnan(d["impact_speed_hit"])


def test_keeping_twenty_centimetres_from_people_is_not_close():
  d = derived(_window(min_dist_person_m=0.25))
  assert d["near_miss"] is False and d["close_person"] is False
