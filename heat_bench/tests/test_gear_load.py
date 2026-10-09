"""Tests for GearLoadObservation and its go2_eval_env_cfg wiring."""

import importlib
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from heat_bench.envs_mjlab.eval_observations import ThermalEnergyObservation
from heat_bench.envs_mjlab.gear_load import GearLoadObservation, gear_load_peak_ratio
from heat_bench.envs_mjlab.go2_eval_env_cfg import (
  go2_eval_env_cfg,
  load_heat_bench_config,
)

from mjlab.asset_zoo.robots import get_go1_robot_cfg
from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg

ASSET_CFG = SceneEntityCfg(name="robot", joint_names=(".*",))


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def setup():
  device = get_test_device()
  entity = Entity(get_go1_robot_cfg())
  model = entity.compile()
  sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
  entity.initialize(model, sim.model, sim.data, device)
  # A no-op step: real physics would overwrite the qfrc_actuator/qacc
  # values the tests write by hand.
  env = Mock()
  env.num_envs = 1
  env.device = device
  env.step_dt = 0.02
  env.scene = {"robot": entity}
  env.sim = SimpleNamespace(model=sim.model, step=lambda: None)
  cfg = ObservationTermCfg(func=GearLoadObservation, params={"asset_cfg": ASSET_CFG})
  term = GearLoadObservation(cfg, env)
  # Raw DOF column of the first joint (the free base occupies the first
  # several DOFs), and its armature and spec effort limit.
  dof = entity.indexing.joint_v_adr[term._joint_ids[0]].item()
  armature = sim.model.dof_armature[0, dof].item()
  return SimpleNamespace(env=env, sim=sim, term=term, dof=dof, armature=armature)


def _substep(s, tau: float, qacc: float = 0.0) -> None:
  s.sim.data.qfrc_actuator[:] = 0.0
  s.sim.data.qacc[:] = 0.0
  s.sim.data.qfrc_actuator[:, s.dof] = tau
  s.sim.data.qacc[:, s.dof] = qacc
  s.env.sim.step()


def test_peak_over_substeps_equals_motor_torque_at_zero_accel(setup):
  s = setup
  for tau in (1.0, -10.0, 1.0):
    _substep(s, tau)
  ratio = s.term(s.env, ASSET_CFG)

  # Peak |tau|, not the last sample or the mean.
  assert s.term.last_peak_gear_load[0, 0].item() == pytest.approx(10.0)
  limit = s.term.effort_limit[0, 0].item()
  assert ratio[0, 0].item() == pytest.approx(10.0 / limit)
  assert s.term._substep_count == 0
  assert s.term._peak_accum.abs().sum().item() == 0.0  # Drained.


def test_rotor_braking_adds_to_gear_load(setup):
  """A joint decelerating against the motor's torque carries more through
  the gears than the motor applies: tau_gear = tau - armature * qacc."""
  s = setup
  assert s.armature > 0.0  # Go1 models reflected rotor inertia.
  tau, qacc = 20.0, -3000.0
  _substep(s, tau, qacc)
  s.term(s.env, ASSET_CFG)

  expected = tau - s.armature * qacc
  assert s.term.last_peak_gear_load[0, 0].item() == pytest.approx(expected, rel=1e-5)
  assert expected > s.term.effort_limit[0, 0].item()  # Beyond the motor's limit.


def test_effort_limit_is_spec_value(setup):
  limits = setup.term.effort_limit[0]
  names = setup.term.joint_names
  for name, limit in zip(names, limits.tolist(), strict=True):
    assert limit == pytest.approx(35.55 if "calf" in name else 23.7)


def test_reset_keeps_final_step_and_metric_reads_live_peak(setup):
  """Auto-reset runs between the decimation loop and observation compute,
  so the episode's final step (often the impact that ended it) must survive
  reset and be visible to the metric before __call__ publishes it."""
  s = setup
  _substep(s, 30.0)
  s.term.reset(torch.tensor([0]))
  limit = s.term.effort_limit[0, 0].item()

  env = Mock()
  env.observation_manager.get_term_cfg.return_value = Mock(func=s.term)
  metric = gear_load_peak_ratio(env, obs_group="gear_load", obs_term="gear_load")
  assert metric[0].item() == pytest.approx(30.0 / limit)

  s.term(s.env, ASSET_CFG)
  assert s.term.last_peak_gear_load[0, 0].item() == pytest.approx(30.0)


def test_chains_with_thermal_substep_hook(setup):
  s = setup
  env = s.env
  env.sim.step = lambda: None
  thermal_cfg = ObservationTermCfg(
    func=ThermalEnergyObservation,
    params={"asset_cfg": ASSET_CFG, "config": load_heat_bench_config()},
  )
  thermal = ThermalEnergyObservation(thermal_cfg, env)
  gear_cfg = ObservationTermCfg(
    func=GearLoadObservation, params={"asset_cfg": ASSET_CFG}
  )
  gear = GearLoadObservation(gear_cfg, env)

  env.sim.step()
  assert thermal._substep_count == 1
  assert gear._substep_count == 1


def _cfg_module():
  # See test_thermal_energy_observation.py: the package re-exports a
  # function shadowing the submodule name.
  return importlib.import_module("heat_bench.envs_mjlab.go2_eval_env_cfg")


def test_gear_load_group_and_metrics_registered_by_default():
  cfg = go2_eval_env_cfg(play=True)
  term = cfg.observations["gear_load"].terms["gear_load"]
  assert term.func is GearLoadObservation
  assert cfg.metrics["gear_load_max_ratio"].reduce == "max"
  assert cfg.metrics["gear_load_mean_peak_ratio"].reduce == "mean"
  # The thermal group is untouched (still a single 14-wide term).
  assert list(cfg.observations["thermal"].terms) == ["thermal_energy"]


def test_gear_load_absent_when_disabled(monkeypatch):
  hb_cfg = load_heat_bench_config()
  hb_cfg["gear_load"]["enabled"] = False
  monkeypatch.setattr(_cfg_module(), "load_heat_bench_config", lambda: hb_cfg)

  cfg = go2_eval_env_cfg(play=True)
  assert "gear_load" not in cfg.observations
  assert "gear_load_max_ratio" not in cfg.metrics
