"""Tests for the Go2 asset and the HIM task built on it."""

import mujoco
import pytest
import torch
from go2_him.him.env_cfg import CRITIC_SLICES, ONE_STEP_OBS_DIM, go2_him_env_cfg
from go2_him.him.modules import frame_major_index
from go2_him.robot.go2_constants import LEGS, get_go2_robot_cfg, get_spec

from mjlab.entity.entity import Entity
from mjlab.terrains.config import flat
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg


def test_spec_is_adapted_to_mjlab():
  model = get_spec().compile()
  assert model.nu == 0 and model.nkey == 0
  names = {model.geom(i).name for i in range(model.ngeom)}
  for leg in LEGS:
    assert {
      f"{leg}_hip_collision",
      f"{leg}_thigh_collision",
      f"{leg}_calf_collision1",
      f"{leg}_calf_collision2",
      f"{leg}_foot_collision",
    } <= names
    assert model.site(leg).id >= 0
  assert {"trunk_collision", "head_collision", "head_lower_collision"} <= names
  assert model.body("trunk").id == 1
  assert all(model.geom_margin == 0)
  sensors = {model.sensor(i).name for i in range(model.nsensor)}
  assert {"imu_ang_vel", "imu_lin_vel"} <= sensors


def test_entity_has_twelve_position_actuators():
  model = Entity(get_go2_robot_cfg()).spec.compile()
  assert model.nu == 12
  assert model.actuator_gainprm[:, 0].tolist() == [20.0] * 12
  joints = [model.joint(i).name for i in range(1, model.njnt)]
  # MJCF body order (FL, FR, RL, RR), not the Unitree SDK motor order (LEGS).
  assert joints == [
    f"{leg}_{j}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for j in ("hip", "thigh", "calf")
  ]


@pytest.mark.slow
def test_env_observations_and_stance():
  """Build the play env on flat ground: observation sizes match HIMLoco,
  the reordered history's newest frame equals the critic's noise-free
  proprioception, and the robot holds its stance on zero actions."""
  from mjlab.envs import ManagerBasedRlEnv

  cfg = go2_him_env_cfg(play=True)
  cfg.scene.num_envs = 4
  cfg.events.pop("randomize_terrain")
  for name in ("payload", "base_com", "pd_gains"):
    cfg.events.pop(name)
  cfg.events["reset_robot_joints"].params["position_range"] = (1.0, 1.0)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(8.0, 8.0), num_rows=2, num_cols=2, sub_terrains={"flat": flat()}
  )
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    obs, _ = env.reset()
    actions = torch.zeros(4, 12, device=device)
    for _ in range(100):
      obs, *_ = env.step(actions)
    actor, critic = obs["actor"], obs["critic"]
    assert isinstance(actor, torch.Tensor) and isinstance(critic, torch.Tensor)
    assert actor.shape[1] == 6 * ONE_STEP_OBS_DIM
    assert critic.shape[1] == CRITIC_SLICES["height"][1]
    index = frame_major_index((3, 3, 3, 12, 12, 12), 6).to(device)
    current = actor[:, index][:, :ONE_STEP_OBS_DIM]
    lo, hi = CRITIC_SLICES["proprio"]
    torch.testing.assert_close(current, critic[:, lo:hi])
    z = env.scene["robot"].data.root_link_pos_w[:, 2]
    assert ((z > 0.22) & (z < 0.32)).all(), z
  finally:
    env.close()


def _flat(cfg) -> None:
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(8.0, 8.0), num_rows=2, num_cols=2, sub_terrains={"flat": flat()}
  )


@pytest.mark.slow
@pytest.mark.parametrize("collision", ["capsule", "original"])
def test_training_env_runs_himloco_terms(collision):
  """The training env (both collision variants): the disturbance shows in
  the critic only on every 8th step, terminal critic observations are
  saved on reset steps, and small planar commands are zeroed."""
  from go2_him.him.mdp import TERMINAL_CRITIC

  from mjlab.envs import ManagerBasedRlEnv

  cfg = go2_him_env_cfg(collision=collision, ccd_iterations=50)
  cfg.scene.num_envs = 8
  cfg.episode_length_s = 0.5  # Time-outs every 25 steps.
  _flat(cfg)
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  try:
    env.reset()
    actions = torch.zeros(8, 12, device=device)
    lo, hi = CRITIC_SLICES["disturbance"]
    saw_terminal = False
    for _ in range(60):
      obs, _, terminated, truncated, extras = env.step(actions)
      critic = obs["critic"]
      assert isinstance(critic, torch.Tensor)
      force = critic[:, lo:hi]
      if env.common_step_counter % 8 == 0:
        assert (force.abs() > 0).any()
      else:
        assert (force == 0).all()
      if bool((terminated | truncated).any()):
        terminal = extras[TERMINAL_CRITIC]
        assert terminal is not None and terminal.shape == critic.shape
        saw_terminal = True
      else:
        assert extras[TERMINAL_CRITIC] is None
      cmd = env.command_manager.get_command("twist")
      assert cmd is not None
      norm = torch.norm(cmd[:, :2], dim=1)
      assert ((norm == 0) | (norm > 0.2)).all()
    assert saw_terminal
  finally:
    env.close()


def test_original_collision_shapes_kept():
  model = get_spec(capsules=False).compile()
  assert model.geom("FL_thigh_collision").type[0] == mujoco.mjtGeom.mjGEOM_BOX
  assert model.geom("FL_calf_collision1").type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER
  assert model.geom("head_collision").type[0] == mujoco.mjtGeom.mjGEOM_CYLINDER


def test_go2_xml_parses_unmodified():
  # The copied MJCF stays loadable on its own (motors, sensors, keyframe).
  from go2_him.robot.go2_constants import GO2_XML

  model = mujoco.MjModel.from_xml_path(str(GO2_XML))
  assert model.nu == 12 and model.nkey == 1


@pytest.mark.slow
def test_free_fault_without_actuator_health():
  """On the Go2 HIM task (no heat_bench thermal stack) a FREE fault zeroes
  the failed joint's motor and a reset restores it; the HIM command sampler
  is replaced by a held uniform command."""
  from heat_bench.envs_mjlab.go2_eval_env_cfg import add_joint_fault_injection
  from heat_bench.scripts.fault_impact import _base_env_cfg

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  cfg = _base_env_cfg("Go2-HIM-Rough")
  add_joint_fault_injection(
    cfg,
    ("FR_calf_joint",),
    modes=("free",),
    fault_time_s=(0.1, 0.1),
    command_vel=(0.5, 0.0, 0.0),
  )
  assert type(cfg.commands["twist"]) is UniformVelocityCommandCfg
  cfg.scene.num_envs = 2
  _flat(cfg)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    env.reset()
    fault = env.event_manager.get_term_cfg("joint_fault_injection").func
    for _ in range(10):
      env.step(torch.zeros(2, 12))
    rows = torch.arange(2)
    ctrl = fault._ctrl_ids[fault.fault_joint[:, 0]]
    forcerange = env.sim.model.actuator_forcerange
    assert fault.faulted.all()
    assert (forcerange[rows, ctrl] == 0).all()
    assert (forcerange[0, fault._ctrl_ids].abs().sum(-1) > 0).sum() == 11
    env.reset()
    assert (forcerange[rows, ctrl, 1] > 40).all()
    cmd = env.command_manager.get_command("twist")
    assert cmd is not None
    torch.testing.assert_close(cmd[:, 0], torch.full((2,), 0.5))
  finally:
    env.close()
