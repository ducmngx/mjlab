"""Household layouts: routes, the built terrain, and the critic's fault set."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from failbench.faults import FaultMixture
from failbench.house import (
  LAYOUTS,
  PROBE_CORRIDOR_WIDTHS,
  PROBE_DOOR_GAPS,
  PROBE_DOOR_X,
  PROBE_MIXED_GAP,
  PROBE_SLOPE_LANES,
  PROBE_SLOPE_OFFSET,
  ground_height,
  probe_lanes,
)
from failbench.house_terrain import HouseTerrainCfg, route_table
from failbench.mppi.risk_critic import (
  REPRESENTATIVE_FAULTS,
  RiskCriticCfg,
  fault_names,
  imagined_faults,
)
from failbench.severity import SEVERITY, geom_severity


def _build(name: str):
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  cfg = HouseTerrainCfg(layout=name)
  out = cfg.function(0.0, spec, np.random.default_rng(0))
  return spec.compile(), out.origin


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_every_route_runs_from_its_start_place_to_its_goal_place(name):
  lay = LAYOUTS[name]
  table, paths = route_table(name)
  assert len(table) == len(paths) == len(lay.routes)
  for (a, b), row, path in zip(lay.routes, table, paths, strict=True):
    assert np.hypot(*(row[:2] - lay.places[a])) < 0.1
    assert np.hypot(*(row[4:6] - lay.places[b])) < 0.1
    assert np.hypot(*(path[-1] - row[4:6])) < 1e-6
    # The heading points along the path's start.
    d = path[min(8, len(path) - 1)] - path[0]
    assert np.cos(np.arctan2(d[1], d[0]) - row[2]) > 0.7


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_the_built_ground_is_the_ground_the_planners_map_is_drawn_from(name):
  lay = LAYOUTS[name]
  model, origin = _build(name)
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  rng = np.random.default_rng(1)
  w, h = lay.size
  pts = rng.uniform([0.3, 0.3], [w - 0.3, h - 0.3], (600, 2))

  def clear(x, y, gap=0.08):
    """Away from every item's edge, and not under an object."""
    for it in lay.items:
      inside = it.x0 - gap < x < it.x1 + gap and it.y0 - gap < y < it.y1 + gap
      core = it.x0 + gap < x < it.x1 - gap and it.y0 + gap < y < it.y1 - gap
      if inside and (it.cls != "floor" or not core):
        return False
    return True

  geomid = np.zeros(1, np.int32)
  checked = 0
  for x, y in pts:
    if not clear(x, y):
      continue
    start = np.array([origin[0] + x, origin[1] + y, 5.0])
    dist = mujoco.mj_ray(
      model, data, start, np.array([0.0, 0.0, -1.0]), None, True, -1, geomid
    )
    want = float(ground_height(lay, np.array(x), np.array(y)))
    assert abs((5.0 - dist) - want) < 0.015, (x, y, 5.0 - dist, want)
    checked += 1
  assert checked > 100


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_every_object_is_priced_as_its_class(name):
  lay = LAYOUTS[name]
  model, _ = _build(name)
  sev = geom_severity(model).tolist()[:-1]
  names = [model.geom(i).name for i in range(model.ngeom)]
  for cls in ("glass", "person", "box"):
    priced = [
      s for n, s in zip(names, sev, strict=True) if n.startswith(f"course_{cls}_")
    ]
    assert len(priced) == sum(it.cls == cls for it in lay.items)
    assert set(priced) <= {SEVERITY[cls]}
  floors = [s for n, s in zip(names, sev, strict=True) if n.startswith("course_floor_")]
  assert floors and set(floors) == {0.0}


def test_the_critic_can_imagine_only_the_eight_representative_faults():
  cfg = RiskCriticCfg(faults=REPRESENTATIVE_FAULTS)
  params, prior = imagined_faults(cfg, 12)
  assert sorted(fault_names(params)) == sorted(REPRESENTATIVE_FAULTS)
  assert len(REPRESENTATIVE_FAULTS) == 8
  assert abs(float(prior.sum()) - 1) < 1e-6
  # Each keeps the mixture's odds against the others.
  mix = FaultMixture()
  _, lock_p, weak_p = mix.probs
  names = fault_names(params)
  lock = prior[names.index("FL_calf_joint:lock")]
  limp = prior[names.index("FL_hip_joint:limp")]
  assert abs(float(lock / limp) - lock_p / (weak_p * mix.limp_prob)) < 1e-4


def _gap_beside(lay, cls: str, x: float, lane: tuple[float, float]) -> float:
  """The free width (m) at ``x`` in ``lane`` between the object of ``cls``
  and the door's other jamb."""
  y0, y1 = lane
  ob = next(
    it for it in lay.items if it.cls == cls and it.x0 < x < it.x1 and y0 < it.y0 < y1
  )
  jamb = min(
    it.y0
    for it in lay.items
    if it.cls == "wall" and it.x0 < x < it.x1 and ob.y1 <= it.y0 < y1
  )
  return jamb - ob.y1


def test_the_door_probes_leave_the_designed_gaps_beside_the_person_or_box():
  for name, cls in (("probe_door", "person"), ("probe_door_box", "box")):
    lay = LAYOUTS[name]
    gaps = [_gap_beside(lay, cls, PROBE_DOOR_X, lane) for lane in probe_lanes(lay)]
    np.testing.assert_allclose(gaps, PROBE_DOOR_GAPS, atol=1e-6)


def test_the_slope_probes_mirror_each_object_downhill_and_uphill():
  lay = LAYOUTS["probe_slope"]
  table, _ = route_table("probe_slope")
  offsets = {}
  for (y0, y1), (cls, side) in zip(probe_lanes(lay), PROBE_SLOPE_LANES, strict=True):
    route_y = (y0 + y1) / 2
    ob = next(it for it in lay.items if it.cls == cls and y0 < it.y0 < y1 and it.x0 > 1)
    near = route_y - ob.y1 if side == "down" else ob.y0 - route_y
    offsets[(cls, side)] = near
    # Downhill is the lower side of the lane's cross slope.
    centre = float(ground_height(lay, np.array(5.0), np.array((ob.y0 + ob.y1) / 2)))
    mid = float(ground_height(lay, np.array(5.0), np.array(route_y)))
    assert (centre < mid) == (side == "down")
  for cls in ("person", "glass"):
    assert offsets[(cls, "down")] == pytest.approx(offsets[(cls, "up")])
    assert offsets[(cls, "down")] == pytest.approx(PROBE_SLOPE_OFFSET)
  assert len(table) == 2 * len(PROBE_SLOPE_LANES)


def test_no_severity_scaled_filter_margin_both_passes_the_door_and_clears_the_slope():
  """The mixed probe's point: with the severity-scaled filter (margin m at
  glass, 2m at people, 0.16m at walls; the Go2 is 0.36 m wide), the door
  beside the person passes only below some m and the person off the slope
  pushes the robot off its route only above a larger m."""
  width = 0.36
  passes_door_below = (PROBE_MIXED_GAP - width) / (2 + 0.16)
  pushes_off_above = (PROBE_SLOPE_OFFSET - width / 2) / 2
  assert passes_door_below < pushes_off_above


@pytest.mark.parametrize("name", ["probe_corridor", "probe_corridor_swap"])
def test_the_corridor_probes_leave_the_designed_width_between_glass_and_wood(name):
  lay = LAYOUTS[name]
  glass_below = name == "probe_corridor"
  for (y0, y1), width in zip(probe_lanes(lay), PROBE_CORRIDOR_WIDTHS, strict=True):
    glass = [it for it in lay.items if it.cls == "glass" and y0 < it.y0 < y1]
    wood = [it for it in lay.items if it.cls == "box" and y0 < it.y0 < y1]
    assert glass and wood
    lo, hi = (glass, wood) if glass_below else (wood, glass)
    free = min(it.y0 for it in hi) - max(it.y1 for it in lo)
    assert free == pytest.approx(width)
    # Centred on the lane, along the same stretch.
    assert max(it.y1 for it in lo) + free / 2 == pytest.approx((y0 + y1) / 2)
    assert {(it.x0, it.x1) for it in glass} == {(it.x0, it.x1) for it in wood}
