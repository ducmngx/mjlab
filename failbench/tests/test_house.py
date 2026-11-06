"""Household layouts: routes, the built terrain, and the critic's fault set."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
from failbench.faults import FaultMixture
from failbench.house import LAYOUTS, ground_height
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
    assert set(priced) == {SEVERITY[cls]}
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
