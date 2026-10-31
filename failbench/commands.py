"""Goal-directed twist commands with an open-loop post-fault plan.

Each trial has a goal (``failbench.goals.sample_goal``). The fault is
known the moment it fires (no detection), and the command is:

- Before the fault: ``warmup_s`` standing still, then a goal follower,
  closed loop. The turn rate is
  proportional to the heading error to the goal, and the forward speed (the
  trial's cruise speed) is scaled by the cosine of that error, so the robot
  turns before it walks. It stops within ``arrive_radius`` of the goal.
  Each trial draws a pre-fault mode (``PrefaultCfg``): it walks so (the
  default), stands until the fault, sets off shortly before it, walks
  slowly, or walks and stops shortly before it, so the impact model also
  sees faults at rest and at low speed, as a planner meets them.
- From the fault: a segment plan (``failbench.plans.SegmentPlan``) started
  from the follower's command on the fault step (``c0``) and the robot's
  pose then, and open loop from there on: a hold of ``c0``, or a few
  segments pulled toward the goal, or with ``lane_heading`` set (the
  course), along the lane centre line through the goal. It never reacts to the drift the impact
  model has to predict, like a candidate sequence MPPI scores.

Where ``use_external`` is set, ``external`` (a planner's command, e.g.
``failbench/scripts/run_mppi.py``) replaces both.

Planar commands with norm at or below ``zero_below`` are zeroed, as in
HIMLoco's sampler (``go2_him/him/commands.py``). Step events run before the
command update in ``ManagerBasedRlEnv.step``, so the plan starts on the
fault step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from failbench.faults import FAULT_TERM
from failbench.plans import PlanCfg, SegmentPlan
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GOAL_TERM = "goal"


PREFAULT_MODES = ("walk", "stand", "start", "slow", "stopped")


@dataclass(frozen=True)
class PrefaultCfg:
  """What a trial does before its fault, timed against the fault time (on
  stairs, whose faults fire by position, the timeout: there ``stopped``
  mostly faults while walking)."""

  mix: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 0.0)
  """Share of each of ``PREFAULT_MODES`` (normalised)."""
  start_lead_s: tuple[float, float] = (0.1, 1.5)
  """``start``: sets off this long before the fault."""
  slow_speed: tuple[float, float] = (0.2, 0.5)
  """``slow``: cruise speed (``zero_below`` zeroes slower commands)."""
  stop_lead_s: tuple[float, float] = (0.3, 2.0)
  """``stopped``: stops this long before the fault."""


def prefault_still(
  t: torch.Tensor,
  fault_time_s: torch.Tensor,
  mode: torch.Tensor,
  lead: torch.Tensor,
  warmup_s: float,
) -> torch.Tensor:
  """Whether each trial's follower stands at time ``t`` (s), by its mode
  (``PREFAULT_MODES``) and ``lead`` (s before ``fault_time_s``)."""
  to_fault = fault_time_s - t
  return (
    (t < warmup_s)
    | (mode == PREFAULT_MODES.index("stand"))
    | ((mode == PREFAULT_MODES.index("start")) & (to_fault > lead))
    | ((mode == PREFAULT_MODES.index("stopped")) & (to_fault <= lead))
  )


class GoalVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg: GoalVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.goal_cfg = cfg
    n, dev = self.num_envs, self.device
    self.speed = torch.zeros(n, device=dev)
    """Cruise speed of the goal follower."""
    self.mode = torch.zeros(n, dtype=torch.long, device=dev)
    """Pre-fault mode, an index into ``PREFAULT_MODES``."""
    self.lead = torch.zeros(n, device=dev)
    """``start`` / ``stopped``: seconds before the fault it sets off / stops."""
    self.plan = SegmentPlan(n, cfg.plan, dev)
    self.c0 = torch.zeros(n, 3, device=dev)
    self.started = torch.zeros(n, dtype=torch.bool, device=dev)
    self.external = torch.zeros(n, 3, device=dev)
    """A command written from outside (a planner), used where
    ``use_external``; still deadbanded."""
    self.use_external = torch.zeros(n, dtype=torch.bool, device=dev)

  def _deadband(self, cmd: torch.Tensor) -> torch.Tensor:
    small = cmd[..., :2].norm(dim=-1, keepdim=True) <= self.goal_cfg.zero_below
    return torch.cat([torch.where(small, 0.0, cmd[..., :2]), cmd[..., 2:]], dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pc, m, dev = self.goal_cfg.prefault, len(env_ids), self.device

    def uniform(r: tuple[float, float]) -> torch.Tensor:
      return r[0] + (r[1] - r[0]) * torch.rand(m, device=dev)

    mix = torch.tensor(pc.mix, device=dev)
    mode = torch.multinomial(mix / mix.sum(), m, replacement=True)
    slow = mode == PREFAULT_MODES.index("slow")
    self.mode[env_ids] = mode
    self.speed[env_ids] = torch.where(
      slow, uniform(pc.slow_speed), uniform(self.goal_cfg.speed)
    )
    self.lead[env_ids] = torch.where(
      mode == PREFAULT_MODES.index("start"),
      uniform(pc.start_lead_s),
      uniform(pc.stop_lead_s),
    )
    self.started[env_ids] = False
    self.use_external[env_ids] = False
    self.vel_command_b[env_ids] = 0.0

  @property
  def goal_w(self) -> torch.Tensor:
    return self._env.event_manager.get_term_cfg(GOAL_TERM).func.goal_w

  def follow(self) -> torch.Tensor:
    """The goal follower's command ``[n, 3]``."""
    c = self.goal_cfg
    to_goal = self.goal_w[:, :2] - self.robot.data.root_link_pos_w[:, :2]
    err = wrap_to_pi(
      torch.atan2(to_goal[:, 1], to_goal[:, 0]) - self.robot.data.heading_w
    )
    wz = (c.heading_gain * err).clamp(-c.max_turn, c.max_turn)
    vx = self.speed * torch.cos(err).clamp(min=0.0)
    cmd = torch.stack([vx, torch.zeros_like(vx), wz], dim=-1)
    arrived = to_goal.norm(dim=-1) < c.arrive_radius
    return torch.where(arrived[:, None], 0.0, cmd)

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids
    fault = self._env.event_manager.get_term_cfg(FAULT_TERM).func
    follow = self.follow()
    running = (fault.faulted & self.started).nonzero().squeeze(-1)
    if len(running):
      self.plan.step(running, self._env.step_dt)
    t = self._env.episode_length_buf.float() * self._env.step_dt
    still = prefault_still(
      t, fault.fault_time_s, self.mode, self.lead, self.goal_cfg.warmup_s
    )
    follow = torch.where(still[:, None], 0.0, follow)
    new = (fault.faulted & ~self.started).nonzero().squeeze(-1)
    if len(new):
      self.c0[new] = follow[new]
      pos = self.robot.data.root_link_pos_w[new, :2]
      yaw = self.robot.data.heading_w[new]
      pose = torch.cat([pos, yaw[:, None]], dim=-1)
      lane_heading = self.goal_cfg.lane_heading
      lane = None if lane_heading is None else torch.ones_like(new, dtype=torch.bool)
      bearing = None if lane_heading is None else torch.full_like(yaw, lane_heading)
      self.plan.start(
        new, follow[new], pose, self.goal_w[new, :2], lane=lane, bearing=bearing
      )
      self.started[new] = True
    cmd = torch.where(fault.faulted[:, None], self.plan.cmd, follow)
    cmd = torch.where(self.use_external[:, None], self.external, cmd)
    self.vel_command_b[:] = self._deadband(cmd)


@dataclass(kw_only=True)
class GoalVelocityCommandCfg(UniformVelocityCommandCfg):
  """``ranges`` is unused."""

  speed: tuple[float, float] = (0.5, 1.0)
  warmup_s: float = 1.0
  """Stand still this long after spawning before following the goal."""
  heading_gain: float = 1.5
  max_turn: float = 1.0
  arrive_radius: float = 0.3
  plan: PlanCfg = field(default_factory=PlanCfg)
  lane_heading: float | None = None
  """World heading (rad) of a lane through the goal for lane-following plans
  (the course: 0, lanes run along +x); ``None`` for goal-bearing plans."""
  zero_below: float = 0.2
  prefault: PrefaultCfg = field(default_factory=PrefaultCfg)

  def build(self, env: ManagerBasedRlEnv) -> GoalVelocityCommand:
    return GoalVelocityCommand(self, env)
