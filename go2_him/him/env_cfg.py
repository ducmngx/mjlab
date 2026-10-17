"""Go2 rough-terrain velocity task with HIMLoco's observations, rewards,
terrain and randomization.

HIMLoco: Long et al., "Hybrid Internal Model: Learning Agile Legged
Locomotion with Simulated Robot Response", ICLR 2024, arXiv:2312.11460;
github.com/InternRobotics/HIMLoco (legged_gym ``legged_robot.py`` and
``legged_robot_config.py``, go1 ``go1_config.py``).

Observations follow HIMLoco's layout exactly, because the estimator slices
the critic observation by position:

- ``actor``: the 45-d one-step proprioception (command, angular velocity,
  projected gravity, joint position, joint velocity, last action), with
  noise, over a 6-step history. mjlab stacks history per term, oldest
  first, so the model reorders it (``modules.frame_major``).
- ``critic``: the same 45 terms (with their own noise draw, as HIMLoco's
  privileged observation shares the noisy proprioception), no history, then
  base linear velocity (the estimator's velocity target), the body-frame
  base disturbance force, and 187 noisy height points; 238-d in total (see
  ``CRITIC_SLICES``).

Everything follows HIMLoco except three settings that trade fidelity for
GPU memory or the robot, which are arguments of ``go2_him_env_cfg``:
``collision`` (Go2 collision shapes), ``ccd_iterations`` and ``gains``.
"""

from __future__ import annotations

import math
from dataclasses import replace

from go2_him.him import mdp as him_mdp
from go2_him.him.commands import HIMVelocityCommandCfg, him_command_curriculum
from go2_him.him.terrains import HIM_TERRAINS_CFG
from go2_him.robot.go2_constants import (
  ACTION_SCALE,
  LEGS,
  Collision,
  Gains,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

HISTORY_LENGTH = 6
CCD_ITERATIONS = 100
"""mujoco_warp sizes its convex-collision (EPA) scratch buffers by this.
mjlab's Go1 rough task uses 500, which needs more than an 8 GB GPU at 2048
envs."""
ONE_STEP_OBS_DIM = 45
"""command 3 + ang vel 3 + gravity 3 + joint pos 12 + joint vel 12 + action 12."""
CRITIC_SLICES = {
  "proprio": (0, 45),
  "lin_vel": (45, 48),
  "disturbance": (48, 51),
  "height": (51, 238),
}
"""Critic observation layout. The estimator's velocity target is
``lin_vel``; its contrastive target is ``[3:48]``, i.e. the proprioception
without the command, plus linear velocity (HIMLoco ``HIMEstimator.update``)."""
NEXT_OBS_SLICE = (3, 48)

# HIMLoco observation scales (legged_gym ``normalization.obs_scales``).
LIN_VEL_SCALE = 2.0
ANG_VEL_SCALE = 0.25
DOF_VEL_SCALE = 0.05
HEIGHT_SCALE = 5.0
COMMAND_SCALE = (LIN_VEL_SCALE, LIN_VEL_SCALE, ANG_VEL_SCALE)

FOOT_GEOMS = tuple(f"{leg}_foot_collision" for leg in LEGS)
TRUNK_GEOMS = ("trunk_collision", "head_collision", "head_lower_collision")


def _proprio_terms() -> dict[str, ObservationTermCfg]:
  """HIMLoco's one-step observation, in its order."""

  def noise(n: float) -> Unoise:
    return Unoise(n_min=-n, n_max=n)

  return {
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
      scale=COMMAND_SCALE,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=noise(0.2),
      scale=ANG_VEL_SCALE,
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity, noise=noise(0.05)
    ),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=noise(0.01)),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=noise(1.5), scale=DOF_VEL_SCALE
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def go2_him_env_cfg(
  play: bool = False,
  collision: Collision = "capsule",
  gains: Gains = "unitree",
  ccd_iterations: int = CCD_ITERATIONS,
) -> ManagerBasedRlEnvCfg:
  """The Go2 HIMLoco task.

  Args:
    play: Viewer variant: no noise, pushes, disturbances or curricula.
    collision: ``"original"`` keeps the Go2 MJCF's cylinder/box collision
      shapes; ``"capsule"`` uses capsules/spheres, which need far less GPU
      memory against heightfields.
    gains: ``"unitree"`` (Kp 20, Kd 0.5, Unitree's Go2 configs) or
      ``"himloco"`` (Kp 40, Kd 1, HIMLoco's Go1 config).
    ccd_iterations: Convex-collision iterations (see ``CCD_ITERATIONS``).
  """
  cfg = make_velocity_env_cfg()

  cfg.sim.nconmax = None
  cfg.sim.mujoco.ccd_iterations = ccd_iterations
  cfg.sim.mujoco.impratio = 10
  cfg.sim.mujoco.cone = "elliptic"
  cfg.sim.contact_sensor_maxmatch = 500

  robot = get_go2_robot_cfg(collision=collision, gains=gains)
  # HIMLoco's action delay: each control step draws d in [0, decimation)
  # and the first d physics substeps still use the previous action. mjlab's
  # actuator delay does exactly that with the lag resampled once per
  # control step, in phase across envs.
  assert robot.articulation is not None
  for actuator in robot.articulation.actuators:
    actuator.delay_min_lag = 0
    actuator.delay_max_lag = cfg.decimation - 1
    actuator.delay_update_period = cfg.decimation
    actuator.delay_per_env_phase = False
  cfg.scene.entities = {"robot": robot}

  ##
  # Sensors.
  ##

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "trunk"
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in LEGS)
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)
  trunk_contact = ContactSensorCfg(
    name="trunk_contact",
    primary=ContactMatch(mode="geom", pattern=TRUNK_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (trunk_contact,)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = HIM_TERRAINS_CFG
  cfg.scene.terrain.max_init_terrain_level = 5

  ##
  # Observations.
  ##

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=_proprio_terms(),
      concatenate_terms=True,
      enable_corruption=True,
      history_length=HISTORY_LENGTH,
      flatten_history_dim=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        **_proprio_terms(),
        "base_lin_vel": ObservationTermCfg(
          func=envs_mdp.base_lin_vel, scale=LIN_VEL_SCALE
        ),
        "disturbance": ObservationTermCfg(func=him_mdp.base_disturbance),
        "height": ObservationTermCfg(
          func=him_mdp.height_measurements,
          params={"sensor_name": "terrain_scan"},
          noise=Unoise(n_min=-0.1, n_max=0.1),
          scale=HEIGHT_SCALE,
        ),
      },
      concatenate_terms=True,
      enable_corruption=True,
    ),
  }

  ##
  # Actions and commands.
  ##

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = ACTION_SCALE

  cfg.commands["twist"] = HIMVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(10.0, 10.0),
    heading_command=True,
    heading_control_stiffness=0.5,
    rel_heading_envs=1.0,
    rel_standing_envs=0.0,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(-1.0, 1.0),
      # legged_gym clips the heading controller's yaw rate to +-1.
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),
    ),
  )

  ##
  # Events: HIMLoco's domain randomization and disturbances.
  ##

  trunk = SceneEntityCfg("robot", body_names=("trunk",))
  cfg.events = {
    # legged_gym ``_reset_root_states``: +-1 m in x/y, no yaw change, base
    # linear and angular velocity +-0.5 per axis.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
        "velocity_range": {
          k: (-0.5, 0.5) for k in ("x", "y", "z", "roll", "pitch", "yaw")
        },
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=him_mdp.reset_joints_by_scale,
      mode="reset",
      params={"position_range": (0.5, 1.5)},
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
        "operation": "abs",
        "axes": [0],
        "ranges": (0.2, 1.25),
        "shared_random": True,
      },
    ),
    "payload": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={"asset_cfg": trunk, "operation": "add", "ranges": (-1.0, 2.0)},
    ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": trunk,
        "operation": "add",
        "ranges": {0: (-0.05, 0.05), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
    # Kp/Kd factors are redrawn at every reset. HIMLoco also samples a motor
    # strength factor but never applies it, so there is none here.
    "pd_gains": EventTermCfg(
      mode="reset",
      func=dr.pd_gains,
      params={"kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1)},
    ),
    # Every 16 s, all envs at once: world x/y base velocity set to U(-1, 1).
    "push_robot": EventTermCfg(
      func=him_mdp.set_root_xy_velocity,
      mode="interval",
      interval_range_s=(16.0, 16.0),
      is_global_time=True,
      params={"velocity_range": (-1.0, 1.0)},
    ),
    # Every 8 control steps, all envs at once: a +-30 N body-frame force.
    "disturbance": EventTermCfg(
      func=him_mdp.HIMDisturbance,
      mode="step",
      params={"asset_cfg": trunk, "force_range": (-30.0, 30.0), "interval": 8},
    ),
    # Last, so it sees this step's state: terminal critic observations.
    "terminal_critic": EventTermCfg(
      func=him_mdp.store_terminal_critic, mode="step", params={}
    ),
  }

  ##
  # Rewards: HIMLoco go1 scales.
  ##

  feet = SceneEntityCfg("robot", site_names=LEGS)
  cfg.rewards = {
    "tracking_lin_vel": RewardTermCfg(
      func=him_mdp.track_lin_vel_xy_exp,
      weight=1.0,
      params={"sigma": 0.25, "command_name": "twist"},
    ),
    "tracking_ang_vel": RewardTermCfg(
      func=him_mdp.track_ang_vel_z_exp,
      weight=0.5,
      params={"sigma": 0.25, "command_name": "twist"},
    ),
    "lin_vel_z": RewardTermCfg(func=him_mdp.lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=him_mdp.ang_vel_xy_l2, weight=-0.05),
    "orientation": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-0.2),
    "base_height": RewardTermCfg(
      func=him_mdp.base_height_l2,
      weight=-1.0,
      params={"target_height": 0.30, "sensor_name": "terrain_scan"},
    ),
    "foot_clearance": RewardTermCfg(
      func=him_mdp.foot_clearance,
      weight=-0.01,
      params={"target_height": -0.20, "asset_cfg": feet},
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    "smoothness": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.01),
    "dof_acc": RewardTermCfg(func=him_mdp.joint_acc_fd_l2, weight=-2.5e-7),
    "joint_power": RewardTermCfg(func=him_mdp.joint_power, weight=-2e-5),
  }

  ##
  # Terminations and curriculum.
  ##

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "base_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": trunk_contact.name, "force_threshold": 1.0},
    ),
  }
  cfg.curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=him_mdp.him_terrain_curriculum, params={"command_name": "twist"}
    ),
    # HIMLoco go1: max_curriculum 2 m/s.
    "command_vel": CurriculumTermCfg(
      func=him_command_curriculum,
      params={"command_name": "twist", "max_curriculum": 2.0},
    ),
  }

  cfg.viewer.body_name = "trunk"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    for name in ("push_robot", "disturbance", "terminal_critic"):
      cfg.events.pop(name, None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )
    gen = cfg.scene.terrain.terrain_generator
    assert gen is not None
    cfg.scene.terrain.terrain_generator = replace(
      gen, curriculum=False, num_rows=5, num_cols=5, border_width=10.0
    )

  return cfg
