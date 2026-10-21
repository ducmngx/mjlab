"""Open-loop segment plans: the post-fault command schedule.

A plan is a ``cmd_vel`` schedule built from a few segments, after the
random-walk driver ZLIK collects its data with (Pokhrel et al.,
arXiv:2602.12385; CTRV / CTRA segments in its
``DataGeneration/random_walk_algorithms.py``), with a pull toward the goal
and HIMLoco's command limits. ``SegmentPlan`` holds a
batch of plans and is stepped once per control step:

- ``start`` (at the fault): the plan begins at the command being executed,
  ``c0``, and from the robot's actual pose, and fixes its goal bearing: the
  direction from that pose to the goal. A ``hold_prob`` share of plans
  are holds: ``c0`` forever (nobody reacts to the fault).
- Every other plan cuts time into segments of exponential length (mean
  ``seg_mean_s``, clipped to ``seg_s``). Each segment draws a target twist:
  vx stops (``stop_prob``), keeps the previous target (``keep_prob``) or
  takes a new speed in ``speed``, never negative; wz is a pull toward the
  goal bearing plus noise, ``goal_gain * e + N(0, wz_std)``; vy is usually
  0.
- vx ramps to its target at up to ``max_accel``; vy and wz ramp over
  ``ramp_s``.
- Lane plans (``start(lane=...)``, the course) follow a feasible reference
  path instead: the line through the goal along the bearing (the lane
  centre). wz is recomputed every step as ``lane_heading_gain * e -
  lane_offset_gain * lateral offset`` plus a per-segment noise offset
  (``lane_wz_std``, ramped like wz), so the plan tends back to the centre
  line while its speed segments slow, stop and wobble it. Obstacle contacts
  then come mostly from the drift a fault causes, not from the plan.

The pull is toward a bearing, not the goal point, so a plan that runs past
the goal keeps heading the same way instead of circling back to it. The
heading error ``e`` is that bearing minus the plan's own predicted yaw, the
commanded turn rate integrated from the start (as ZLIK's driver integrates
its own yaw), never the robot's measured heading: a plan is fixed by its
start state and random draws alone, like a candidate MPPI scores, and so
can't encode the drift the impact model predicts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import wrap_to_pi


@dataclass(frozen=True)
class PlanCfg:
  hold_prob: float = 0.3
  seg_mean_s: float = 1.5
  seg_s: tuple[float, float] = (0.5, 3.0)
  stop_prob: float = 0.15
  keep_prob: float = 0.25
  speed: tuple[float, float] = (0.2, 1.2)
  goal_gain: float = 1.0
  wz_std: float = 0.4
  vy_prob: float = 0.3
  vy_std: float = 0.15
  max_accel: float = 1.0
  ramp_s: float = 0.5
  limits: tuple[float, float, float] = (1.5, 0.5, 1.0)
  """vx in [0, limits[0]], |vy| and |wz| at most limits[1] and limits[2]."""
  lane_heading_gain: float = 1.0
  lane_offset_gain: float = 1.0
  """Turn rate (rad/s) per metre off the lane centre line."""
  lane_wz_std: float = 0.15
  lane_vy_std: float = 0.05


class SegmentPlan:
  """A batch of ``n`` plans; ``cmd`` is each plan's current command."""

  def __init__(self, n: int, cfg: PlanCfg, device: torch.device | str):
    self.cfg = cfg
    self.device = device
    z = lambda *s: torch.zeros(n, *s, device=device)  # noqa: E731
    self.hold = torch.zeros(n, dtype=torch.bool, device=device)
    self.cmd = z(3)
    self.target = z(3)
    self.ramp_from = z(3)
    self.seg_left = z()
    self.seg_count = torch.zeros(n, dtype=torch.long, device=device)
    """Segments started so far (0 for a hold)."""
    self.pose = z(3)
    """Predicted (x, y, yaw) in the world frame: the commanded path."""
    self.bearing = z()
    """World direction to the goal from the start pose (lane plans: the
    lane direction)."""
    self.lane = torch.zeros(n, dtype=torch.bool, device=device)
    self.line = z(2)
    """A point on the lane centre line (the goal)."""
    self.lane_off = z()
    """Lane plans: the current, ramped wz noise offset."""

  def start(
    self,
    ids: torch.Tensor,
    c0: torch.Tensor,
    pose: torch.Tensor,
    goal: torch.Tensor,
    hold: torch.Tensor | None = None,
    lane: torch.Tensor | None = None,
    bearing: torch.Tensor | None = None,
  ) -> None:
    """Begin plans ``ids`` at command ``c0 [m, 3]``, pose ``(x, y, yaw)
    [m, 3]``, goal ``[m, 2]``. ``hold`` overrides the random hold draw;
    ``lane`` marks lane plans, whose centre line runs through the goal along
    ``bearing`` (default: the direction from the pose to the goal)."""
    m = len(ids)
    if hold is None:
      hold = torch.rand(m, device=self.device) < self.cfg.hold_prob
    self.hold[ids] = hold
    self.cmd[ids] = c0
    self.target[ids] = c0
    self.ramp_from[ids] = c0
    self.seg_left[ids] = 0.0
    self.seg_count[ids] = 0
    self.pose[ids] = pose
    if bearing is None:
      to_goal = goal - pose[:, :2]
      bearing = torch.atan2(to_goal[:, 1], to_goal[:, 0])
    self.bearing[ids] = bearing
    self.lane[ids] = False if lane is None else lane
    self.line[ids] = goal
    self.lane_off[ids] = 0.0

  def _new_segments(self, ids: torch.Tensor) -> None:
    c, m, dev = self.cfg, len(ids), self.device
    err = wrap_to_pi(self.bearing[ids] - self.pose[ids, 2])

    u = torch.rand(m, device=dev)
    new_speed = c.speed[0] + (c.speed[1] - c.speed[0]) * torch.rand(m, device=dev)
    vx = torch.where(u < c.stop_prob + c.keep_prob, self.target[ids, 0], new_speed)
    vx = torch.where(u < c.stop_prob, 0.0, vx)
    lane = self.lane[ids]
    noise = torch.randn(m, device=dev)
    wz = torch.where(lane, noise * c.lane_wz_std, c.goal_gain * err + noise * c.wz_std)
    side = torch.rand(m, device=dev) < c.vy_prob
    vy_std = torch.where(lane, c.lane_vy_std, c.vy_std)
    vy = torch.where(side, torch.randn(m, device=dev) * vy_std, 0.0)
    target = torch.stack([vx, vy, wz], dim=-1)
    lim = torch.tensor(c.limits, device=dev)
    target = torch.maximum(torch.minimum(target, lim), -lim)
    target[:, 0] = target[:, 0].clamp(min=0.0)
    self.target[ids] = target
    self.ramp_from[ids] = self.cmd[ids]
    self.ramp_from[ids, 2] = torch.where(lane, self.lane_off[ids], self.cmd[ids, 2])
    length = -c.seg_mean_s * torch.log(torch.rand(m, device=dev).clamp(min=1e-6))
    self.seg_left[ids] += length.clamp(*c.seg_s)
    self.seg_count[ids] += 1

  def step(self, ids: torch.Tensor, dt: float) -> None:
    """Advance plans ``ids`` by ``dt``: integrate the predicted pose under
    the current command, then move the command along the schedule."""
    c = self.cfg
    cmd, yaw = self.cmd[ids], self.pose[ids, 2]
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    self.pose[ids, 0] += (cy * cmd[:, 0] - sy * cmd[:, 1]) * dt
    self.pose[ids, 1] += (sy * cmd[:, 0] + cy * cmd[:, 1]) * dt
    self.pose[ids, 2] = wrap_to_pi(yaw + cmd[:, 2] * dt)

    ids = ids[~self.hold[ids]]
    if not len(ids):
      return
    due = ids[self.seg_left[ids] <= 0.0]
    if len(due):
      self._new_segments(due)
    cmd, target = self.cmd[ids], self.target[ids]
    dvx = (target[:, 0] - cmd[:, 0]).clamp(-c.max_accel * dt, c.max_accel * dt)
    rate = (target[:, 1:] - self.ramp_from[ids, 1:]).abs() / c.ramp_s * dt
    d = target[:, 1:] - cmd[:, 1:]
    cmd = torch.cat([cmd[:, :1] + dvx[:, None], cmd[:, 1:] + d.clamp(-rate, rate)], -1)
    self.cmd[ids] = cmd
    self.seg_left[ids] -= dt

    lane = ids[self.lane[ids]]
    if len(lane):
      # The wz ramp above ran on the command; for lane plans it moves the
      # noise offset instead, and wz is the centre-line pull plus it.
      off = self.lane_off[lane]
      step = (self.target[lane, 2] - self.ramp_from[lane, 2]).abs() / c.ramp_s * dt
      off = off + (self.target[lane, 2] - off).clamp(-step, step)
      self.lane_off[lane] = off
      b, pose = self.bearing[lane], self.pose[lane]
      rel = pose[:, :2] - self.line[lane]
      lateral = -torch.sin(b) * rel[:, 0] + torch.cos(b) * rel[:, 1]
      err = wrap_to_pi(b - pose[:, 2])
      wz = c.lane_heading_gain * err - c.lane_offset_gain * lateral + off
      self.cmd[lane, 2] = wz.clamp(-c.limits[2], c.limits[2])
