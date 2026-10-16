"""Tests for the scripted mid-run joint faults (inject_joint_fault)."""

import math
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from heat_bench.envs_mjlab.actuator_health import ActuatorState
from heat_bench.envs_mjlab.go2_eval_env_cfg import (
  add_joint_fault_injection,
  go2_eval_env_cfg,
)
from heat_bench.envs_mjlab.joint_fault_injection import (
  FAULT_CONTACT_SENSOR,
  FaultMode,
  contacts_in_base_frame,
  inject_joint_fault,
)

from mjlab.envs import ManagerBasedRlEnv


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def test_config_strips_resets_and_orders_events():
  cfg = go2_eval_env_cfg(play=True, gear_break=True)
  add_joint_fault_injection(cfg, ("FR_calf_joint",), command_vel=(0.5, 0.0, 0.0))
  names = list(cfg.events)
  assert names.index("joint_fault_injection") < names.index("actuator_health")
  assert cfg.events["joint_fault_injection"].func is inject_joint_fault
  assert cfg.terminations == {}
  assert "gear_obs_group" not in cfg.events["actuator_health"].params
  assert any(s.name == FAULT_CONTACT_SENSOR for s in cfg.scene.sensors or ())
  assert cfg.commands["twist"].ranges.lin_vel_x == (0.5, 0.5)  # type: ignore[attr-defined]


def test_config_rejects_unknown_mode():
  with pytest.raises(ValueError, match="Unknown fault mode"):
    add_joint_fault_injection(go2_eval_env_cfg(play=True), ("FR_calf_joint",), ("jam",))


def test_contacts_in_base_frame_rotates_into_trunk_frame():
  # Trunk at (1, 2, 0.3), yawed +90 deg: world +y is trunk-forward.
  half = math.sqrt(0.5)
  asset = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[1.0, 2.0, 0.3]]),
      root_link_quat_w=torch.tensor([[half, 0.0, 0.0, half]]),
    )
  )
  # One filled slot 0.5 m ahead of the trunk on the ground, one empty slot.
  data = SimpleNamespace(
    pos=torch.tensor([[[1.0, 2.5, 0.0], [0.0, 0.0, 0.0]]]),
    # Sensor force is the robot's push on the terrain (downward, backward).
    force=torch.tensor([[[0.0, -10.0, -100.0], [0.0, 0.0, 0.0]]]),
    normal=torch.tensor([[[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]]]),
  )
  pos_b, force_b, _, force_w, valid = contacts_in_base_frame(
    Mock(data=data),
    asset,  # type: ignore[arg-type]
  )
  assert valid.tolist() == [[True, False]]
  torch.testing.assert_close(pos_b[0, 0], torch.tensor([0.5, 0.0, -0.3]))
  torch.testing.assert_close(force_w[0, 0], torch.tensor([0.0, 10.0, 100.0]))
  torch.testing.assert_close(force_b[0, 0], torch.tensor([10.0, 0.0, 100.0]))


@pytest.mark.slow
def test_each_mode_and_reset_in_real_env():
  cfg = go2_eval_env_cfg(play=True)
  add_joint_fault_injection(
    cfg,
    ("FR_calf_joint",),
    modes=("limit", "lock", "free", "none"),
    fault_time_s=(0.2, 0.2),
  )
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    fault = env.event_manager.get_term_cfg("joint_fault_injection").func
    health = env.event_manager.get_term_cfg("actuator_health").func
    # One env per mode.
    fault.fault_mode[:] = torch.tensor(
      [FaultMode.LIMIT, FaultMode.LOCK, FaultMode.FREE, FaultMode.NONE],
      device=env.device,
    )
    robot = env.scene["robot"]
    j = robot.joint_names.index("FR_calf_joint")
    jnt = fault._global_jnt_ids[j]
    actions = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(60):
      env.step(actions)

    assert fault.faulted.all()
    q_f = fault.fault_qpos[:, 0]
    q = robot.data.joint_pos[:, j]
    jnt_range = env.sim.model.jnt_range[:, jnt]
    # LIMIT: range narrowed around q_f; joint stays inside (soft-limit slack).
    torch.testing.assert_close(
      jnt_range[0, 1] - jnt_range[0, 0], torch.tensor(0.3, device=env.device)
    )
    assert jnt_range[0, 0] - 0.01 <= q[0] <= jnt_range[0, 1] + 0.01
    # LOCK: held at the fault-time angle.
    assert abs(q[1] - q_f[1]) < 0.01
    # FREE: motor limp, range untouched.
    assert health.derate[2, j] == 0.0
    torch.testing.assert_close(jnt_range[2], fault.nominal_jnt_range[2, j])
    # NONE: control trial, recorded but nothing changes.
    assert health.derate[3, j] == 1.0
    torch.testing.assert_close(jnt_range[3], fault.nominal_jnt_range[3, j])
    assert health.state[:, j].tolist() == [
      ActuatorState.LIMITED,
      ActuatorState.LOCKED,
      ActuatorState.FREE,
      ActuatorState.HEALTHY,
    ]

    env.reset()
    assert not fault.faulted.any()
    torch.testing.assert_close(
      env.sim.model.jnt_range[:, fault._global_jnt_ids], fault.nominal_jnt_range
    )
    assert (health.state == ActuatorState.HEALTHY).all()
  finally:
    env.close()
