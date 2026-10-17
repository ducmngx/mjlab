"""Observation, reward and event terms that follow HIMLoco's legged_gym
definitions where mjlab has no equivalent.

Reward functions return the raw per-env term; the reward manager multiplies
by the weight and ``step_dt``, as legged_gym does with its scales.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.sensor import RayCastSensor

_ROBOT = SceneEntityCfg("robot")


##
# Observations.
##


DISTURBANCE_ATTR = "_him_disturbance_b"


def base_disturbance(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The base disturbance force HIMLoco's critic observes: the body-frame
  force sampled this step by ``HIMDisturbance`` (zero on other steps), in N."""
  force = getattr(env, DISTURBANCE_ATTR, None)
  if force is None:
    return torch.zeros(env.num_envs, 3, device=env.device)
  return force


def height_measurements(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  offset: float = 0.5,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """legged_gym's height points: ``clip(base_z - offset - terrain_z, -1, 1)``.

  Rays that miss read as terrain at the ray origin, which clips to -1 like a
  very tall obstacle; the scan starts well above the trunk, so this only
  happens off the terrain.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: RayCastSensor = env.scene[sensor_name]
  base_z = asset.data.root_link_pos_w[:, 2:3]
  terrain_z = sensor.data.hit_pos_w[..., 2]
  return torch.clip(base_z - offset - terrain_z, -1.0, 1.0)


##
# Rewards.
##


def track_lin_vel_xy_exp(
  env: ManagerBasedRlEnv, sigma: float, command_name: str, asset_cfg=_ROBOT
) -> torch.Tensor:
  """``exp(-|cmd_xy - v_xy|^2 / sigma)`` in the base frame."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  err = torch.sum(
    torch.square(cmd[:, :2] - asset.data.root_link_lin_vel_b[:, :2]), dim=1
  )
  return torch.exp(-err / sigma)


def track_ang_vel_z_exp(
  env: ManagerBasedRlEnv, sigma: float, command_name: str, asset_cfg=_ROBOT
) -> torch.Tensor:
  """``exp(-(cmd_wz - wz)^2 / sigma)``."""
  asset: Entity = env.scene[asset_cfg.name]
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  err = torch.square(cmd[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-err / sigma)


def lin_vel_z_l2(env: ManagerBasedRlEnv, asset_cfg=_ROBOT) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRlEnv, asset_cfg=_ROBOT) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def base_height_l2(
  env: ManagerBasedRlEnv, target_height: float, sensor_name: str, asset_cfg=_ROBOT
) -> torch.Tensor:
  """``(mean(base_z - terrain_z) - target)^2`` over the height-scan points."""
  asset: Entity = env.scene[asset_cfg.name]
  sensor: RayCastSensor = env.scene[sensor_name]
  base_z = asset.data.root_link_pos_w[:, 2:3]
  height = torch.mean(base_z - sensor.data.hit_pos_w[..., 2], dim=1)
  return torch.square(height - target_height)


def foot_clearance(
  env: ManagerBasedRlEnv, target_height: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """HIMLoco's swing-height term: for each foot,
  ``(z_foot_b - target)^2 * |v_foot_b,xy|``, summed.

  Foot height and velocity are in the base frame, so ``target_height`` is
  negative (below the base). ``asset_cfg.site_names`` selects the feet.
  """
  asset: Entity = env.scene[asset_cfg.name]
  root_pos = asset.data.root_link_pos_w[:, None, :]
  root_quat = asset.data.root_link_quat_w[:, None, :]
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids] - root_pos
  foot_vel = (
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
    - (asset.data.root_link_lin_vel_w[:, None, :])
  )
  quat = root_quat.expand(-1, foot_pos.shape[1], -1)
  foot_pos_b = quat_apply_inverse(quat, foot_pos)
  foot_vel_b = quat_apply_inverse(quat, foot_vel)
  height_err = torch.square(foot_pos_b[..., 2] - target_height)
  lateral_speed = torch.norm(foot_vel_b[..., :2], dim=-1)
  return torch.sum(height_err * lateral_speed, dim=1)


def joint_power(env: ManagerBasedRlEnv, asset_cfg=_ROBOT) -> torch.Tensor:
  """``sum |torque * joint_vel|``."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(
    torch.abs(asset.data.actuator_force * asset.data.joint_vel[:, asset_cfg.joint_ids]),
    dim=1,
  )


class joint_acc_fd_l2:
  """``sum(((v_prev - v) / dt)^2)`` with the previous control step's joint
  velocities (legged_gym ``_reward_dof_acc``)."""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _ROBOT).name]
    self._asset = asset
    self._last = asset.data.joint_vel.clone()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self._last[ids] = self._asset.data.joint_vel[ids]

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg=_ROBOT) -> torch.Tensor:
    vel = self._asset.data.joint_vel
    acc = (self._last - vel) / env.step_dt
    self._last = vel.clone()
    return torch.sum(torch.square(acc), dim=1)


##
# Curriculum.
##


def him_terrain_curriculum(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor, command_name: str, asset_cfg=_ROBOT
) -> dict[str, torch.Tensor]:
  """legged_gym ``_update_terrain_curriculum``: move up after walking more
  than half a tile from the start, down after walking less than half the
  commanded distance over a full episode; past the top, a random level."""
  terrain = env.scene.terrain
  assert terrain is not None and terrain.cfg.terrain_generator is not None
  if env.common_step_counter == 0 or not isinstance(env_ids, torch.Tensor):
    return {"mean": terrain.terrain_levels.float().mean()}
  asset: Entity = env.scene[asset_cfg.name]
  term = env.command_manager.get_term(command_name)
  assert isinstance(term, UniformVelocityCommand)
  start = term.episode_start_pos_w
  distance = torch.norm(asset.data.root_link_pos_w[env_ids, :2] - start[env_ids], dim=1)
  cmd = env.command_manager.get_command(command_name)
  assert cmd is not None
  tile = terrain.cfg.terrain_generator.size[0]
  move_up = distance > tile / 2
  move_down = (
    distance < torch.norm(cmd[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  ) & ~move_up
  terrain.update_env_origins(env_ids, move_up, move_down)
  return {"mean": terrain.terrain_levels.float().mean()}


##
# Events.
##


def set_root_xy_velocity(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Overwrite the world-frame base x/y velocity with ``U(velocity_range)``
  (legged_gym ``_push_robots``)."""
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  vel = asset.data.root_link_vel_w[env_ids].clone()
  vel[:, :2] = torch.empty_like(vel[:, :2]).uniform_(*velocity_range)
  asset.write_root_link_velocity_to_sim(vel, env_ids=env_ids)


class HIMDisturbance:
  """legged_gym ``_disturbance_robots``: every ``interval`` control steps
  (all envs together), a body-frame force ``U(force_range)`` per axis on the
  trunk, visible to the critic (``base_disturbance``) on the step it is drawn.

  Isaac Gym applies that force for one physics substep. mjlab's step events
  can only hold a wrench for a whole control step, so the force is divided
  by the decimation (same impulse, spread over the control step) while the
  observation reports the full sampled force, as in HIMLoco.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._force_b = torch.zeros(env.num_envs, 3, device=env.device)
    self._applied = False
    setattr(env, DISTURBANCE_ATTR, self._force_b)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    pass

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    force_range: tuple[float, float],
    interval: int,
  ) -> None:
    del env_ids, asset_cfg
    self._force_b.zero_()
    zeros = torch.zeros(env.num_envs, 1, 3, device=env.device)
    if env.common_step_counter % interval == 0:
      self._force_b.uniform_(*force_range)
      quat = self._asset.data.root_link_quat_w
      force_w = quat_apply(quat, self._force_b) / env.cfg.decimation
      self._asset.write_external_wrench_to_sim(
        force_w.unsqueeze(1), zeros, body_ids=self._body_ids
      )
      self._applied = True
    elif self._applied:
      self._asset.write_external_wrench_to_sim(zeros, zeros, body_ids=self._body_ids)
      self._applied = False


TERMINAL_CRITIC = "him_terminal_critic"


def store_terminal_critic(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, group: str = "critic"
) -> None:
  """Save the ``group`` observation of envs about to reset, before mjlab
  resets them, in ``env.extras[TERMINAL_CRITIC]`` (``None`` when no env
  resets). Step events run after termination checks and before auto-reset,
  so this is the terminal observation HIMLoco's estimator trains on
  (legged_gym ``compute_termination_observations``)."""
  del env_ids
  if not bool(env.reset_buf.any()):
    env.extras[TERMINAL_CRITIC] = None
    return
  obs = env.observation_manager.compute_group(group, update_history=False)
  assert isinstance(obs, torch.Tensor)
  env.extras[TERMINAL_CRITIC] = obs.clone()


def reset_joints_by_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> None:
  """Reset joints to the default pose times a per-joint factor in
  ``position_range`` (legged_gym ``_reset_dofs``), clamped to the soft
  limits, with zero velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  default_pos = asset.data.default_joint_pos
  limits = asset.data.soft_joint_pos_limits
  assert default_pos is not None and limits is not None
  pos = default_pos[env_ids].clone()
  lo, hi = position_range
  pos *= torch.empty_like(pos).uniform_(lo, hi)
  pos = pos.clamp(limits[env_ids, :, 0], limits[env_ids, :, 1])
  asset.write_joint_state_to_sim(pos, torch.zeros_like(pos), env_ids=env_ids)
