"""Per-trial goals that fit the terrain, and a spawn heading that faces them.

``sample_goal`` is a reset event, run after the base reset, that sets each
env's goal (``goal_w``) by terrain kind (``GoalMode``):

- ``pyramid`` (stairs, stairs_up): a point on the tile's flat outer border,
  ``pyramid_goal_cheb`` m (Chebyshev) from the tile centre in a random
  direction, so the path from the centre platform or pit crosses the whole
  flight (down on ``stairs``, up on ``stairs_up``).
- ``free`` (rough): ``free_goal_distance`` m from the spawn in a random
  direction.
- ``lane`` (course): the lane end, ``lane_goal_x`` m down the lane from the
  spawn origin.

For ``pyramid`` and ``free`` the robot is turned to face its goal; on the
course it keeps the lane-facing spawn jitter.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

GoalMode = Literal["pyramid", "free", "lane"]


class sample_goal:
  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
    self._asset = env.scene[asset_cfg.name]
    self.goal_w = torch.zeros(env.num_envs, 3, device=env.device)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    mode: GoalMode,
    pyramid_goal_cheb: float = 3.7,
    free_goal_distance: float = 5.0,
    lane_goal_x: float = 13.0,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    del asset_cfg
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    m, dev = len(env_ids), env.device
    origin = env.scene.env_origins[env_ids]
    # Read the pose from qpos: body poses are stale until the next forward.
    adr = self._asset.indexing.free_joint_q_adr
    pose = self._asset.data.data.qpos[env_ids[:, None], adr]
    goal = origin.clone()
    if mode == "lane":
      goal[:, 0] += lane_goal_x
      self.goal_w[env_ids] = goal
      return
    phi = (torch.rand(m, device=dev) * 2 - 1) * math.pi
    u = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    if mode == "pyramid":
      r = pyramid_goal_cheb / u.abs().max(dim=-1).values
      goal[:, :2] += r[:, None] * u
    else:
      goal[:, :2] = pose[:, :2] + free_goal_distance * u
    self.goal_w[env_ids] = goal
    to_goal = goal[:, :2] - pose[:, :2]
    yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    zero = torch.zeros_like(yaw)
    quat = torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)
    self._asset.write_root_link_pose_to_sim(
      torch.cat([pose[:, :3], quat], dim=-1), env_ids=env_ids
    )
