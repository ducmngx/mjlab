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
- ``waypoints`` (``l_course``, ``doors``): points relative to the spawn
  origin (corner or doorways, then the goal). ``goal_w`` is the current
  one; ``advance`` moves on once the robot is within a radius of it or has
  passed it, and ``final`` says whether the current one is the last.

- ``route`` (the household layouts, ``failbench/house_terrain.py``): robot
  ``i`` takes route ``i % len(routes)``; it is placed at the route's start
  (raised by the ground there), facing along its global path, and its goal
  is the route's end.

For ``pyramid`` and ``free`` the robot is turned to face its goal; on the
course it keeps the lane-facing spawn jitter.

``route_progress`` measures how far along its route (spawn, then the
waypoints) a robot has got, for faults that fire at a place on the route.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

GoalMode = Literal["pyramid", "free", "lane", "waypoints", "route"]


class sample_goal:
  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
    self._asset = env.scene[asset_cfg.name]
    n = env.num_envs
    self.goal_w = torch.zeros(n, 3, device=env.device)
    wps = cfg.params.get("waypoints") or [[0.0, 0.0]]
    self.waypoints = torch.zeros(n, len(wps), 3, device=env.device)
    """Every waypoint (world), last = the final goal."""
    self.index = torch.zeros(n, dtype=torch.long, device=env.device)
    self.start = torch.zeros(n, 3, device=env.device)
    """Where the first leg starts (the spawn origin)."""

  @property
  def final(self) -> torch.Tensor:
    return self.index == self.waypoints.shape[1] - 1

  def advance(self, pos_xy: torch.Tensor, radius: float) -> None:
    """Move to the next waypoint where ``pos_xy [n, 2]`` is within
    ``radius`` of the current one, or past it: beyond the line through it
    square to the leg that leads to it. Without the second test, a robot
    that goes by just outside the radius (or whose odometry says so) is
    left with a waypoint behind it."""
    rows = torch.arange(len(self.index), device=self.index.device)
    cur = self.goal_w[:, :2]
    prev = torch.where(
      (self.index > 0)[:, None],
      self.waypoints[rows, (self.index - 1).clamp(min=0), :2],
      self.start[:, :2],
    )
    near = (pos_xy - cur).norm(dim=-1) < radius
    passed = ((pos_xy - cur) * (cur - prev)).sum(-1) > 0
    self.index = torch.where((near | passed) & ~self.final, self.index + 1, self.index)
    self.goal_w = self.waypoints[rows, self.index]

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    mode: GoalMode,
    pyramid_goal_cheb: float = 3.7,
    free_goal_distance: float = 5.0,
    lane_goal_x: float = 13.0,
    waypoints: list | None = None,
    routes: list | None = None,
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
    self.start[env_ids] = origin
    if mode == "waypoints":
      assert waypoints is not None
      rel = torch.tensor(waypoints, dtype=torch.float32, device=dev)  # [K, 2]
      wp = origin[:, None].repeat(1, len(rel), 1)
      wp[..., :2] += rel
      self.waypoints[env_ids] = wp
      self.index[env_ids] = 0
      self.goal_w[env_ids] = wp[:, 0]
      return
    if mode == "route":
      assert routes is not None
      table = torch.tensor(routes, dtype=torch.float32, device=dev)  # [R, 6]
      row = table[env_ids % len(table)]
      goal[:, :2] += row[:, 4:6]
      self.goal_w[env_ids] = goal
      self.waypoints[env_ids] = goal[:, None]
      xyz = torch.cat([origin[:, :2] + row[:, :2], pose[:, 2:3] + row[:, 3:4]], dim=-1)
      self.start[env_ids] = torch.cat([xyz[:, :2], origin[:, 2:]], dim=-1)
      yaw = row[:, 2]
      zero = torch.zeros_like(yaw)
      quat = torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)
      self._asset.write_root_link_pose_to_sim(
        torch.cat([xyz, quat], dim=-1), env_ids=env_ids
      )
      return
    if mode == "lane":
      goal[:, 0] += lane_goal_x
      self.goal_w[env_ids] = goal
      self.waypoints[env_ids] = goal[:, None]
      return
    phi = (torch.rand(m, device=dev) * 2 - 1) * math.pi
    u = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    if mode == "pyramid":
      r = pyramid_goal_cheb / u.abs().max(dim=-1).values
      goal[:, :2] += r[:, None] * u
    else:
      goal[:, :2] = pose[:, :2] + free_goal_distance * u
    self.goal_w[env_ids] = goal
    self.waypoints[env_ids] = goal[:, None]
    to_goal = goal[:, :2] - pose[:, :2]
    yaw = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    zero = torch.zeros_like(yaw)
    quat = torch.stack([torch.cos(yaw / 2), zero, zero, torch.sin(yaw / 2)], dim=-1)
    self._asset.write_root_link_pose_to_sim(
      torch.cat([pose[:, :3], quat], dim=-1), env_ids=env_ids
    )


def route_progress(
  xy: torch.Tensor, start: torch.Tensor, waypoints: torch.Tensor
) -> torch.Tensor:
  """Distance along each route (``start [n, 2]``, then ``waypoints [n, W,
  2]``) to the point on it nearest ``xy [n, 2]``, ``[n]``: how far a robot
  has got, however it wandered to get there."""
  pts = torch.cat([start[:, None], waypoints], 1)  # [n, W + 1, 2]
  a, b = pts[:, :-1], pts[:, 1:]
  seg = b - a
  length = seg.norm(dim=-1)  # [n, W]
  u = ((xy[:, None] - a) * seg).sum(-1) / length.square().clamp(min=1e-9)
  u = u.clamp(0, 1)
  gap = (a + u[..., None] * seg - xy[:, None]).norm(dim=-1)
  before = length.cumsum(-1) - length
  near = gap.argmin(-1, keepdim=True)
  return (before + u * length).gather(1, near)[:, 0]
