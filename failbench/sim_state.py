"""Snapshot and restore everything a FailBench env carries between steps, so
a run can be forked at any moment (``failbench/scripts/replay_faults.py``).

A step's outcome depends on more than the robot's pose: the policy reads a
history of observations, and the actuators serve the policy's targets
through a delay buffer with per-env lags. A snapshot holds, per env:

- the robot's root pose and velocity (world) and joint positions and
  velocities;
- every observation history buffer (``ObservationManager``) and every delay
  buffer (observation and actuator), in chronological order, with their push
  counts and the delay buffers' lags and step counts;
- the action manager's last three actions;
- the observations the policy will act on next (what ``step`` returned).

``restore`` writes chosen snapshot rows into chosen envs of an env built the
same way (task, terrain), without touching the others. Faults, commands and
the env's clock are the caller's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.buffers.circular_buffer import CircularBuffer
from mjlab.utils.buffers.delay_buffer import DelayBuffer

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _delay_buffers(env: ManagerBasedRlEnv) -> dict[str, DelayBuffer]:
  out: dict[str, DelayBuffer] = {}
  om = env.observation_manager
  for group, terms in om._group_obs_term_delay_buffer.items():
    for term, buf in terms.items():
      out[f"obsdelay.{group}.{term}"] = buf
  seen: set[int] = set()
  for name in env.scene.entities:
    for k, act in enumerate(env.scene[name].actuators):
      buf = getattr(act, "_delay_buffer", None)
      if isinstance(buf, DelayBuffer) and id(buf) not in seen:
        seen.add(id(buf))
        out[f"actdelay.{name}.{k}"] = buf
  return out


def _circular_buffers(env: ManagerBasedRlEnv) -> dict[str, CircularBuffer]:
  out: dict[str, CircularBuffer] = {}
  om = env.observation_manager
  for group, terms in om._group_obs_term_history_buffer.items():
    for term, buf in terms.items():
      out[f"hist.{group}.{term}"] = buf
  for name, delay in _delay_buffers(env).items():
    out[name] = delay._buffer
  return out


def _robot(env: ManagerBasedRlEnv):
  return env.scene["robot"]


def snapshot(
  env: ManagerBasedRlEnv, obs: dict[str, torch.Tensor]
) -> dict[str, np.ndarray]:
  """Every env's state now ``{name: [n, ...]}`` (numpy, float32/int64)."""
  robot = _robot(env)
  d = robot.data
  out = {
    "root_pose": d.root_link_pose_w,
    "root_vel": d.root_link_vel_w,
    "joint_pos": d.joint_pos,
    "joint_vel": d.joint_vel,
    "action": env.action_manager._action,
    "prev_action": env.action_manager._prev_action,
    "prev_prev_action": env.action_manager._prev_prev_action,
  }
  for key, value in obs.items():
    out[f"obs.{key}"] = value
  for name, buf in _circular_buffers(env).items():
    if buf._buffer is None:
      continue
    out[f"{name}.data"] = buf.buffer
    out[f"{name}.pushes"] = buf._num_pushes
  for name, delay in _delay_buffers(env).items():
    out[f"{name}.lags"] = delay._current_lags
    out[f"{name}.steps"] = delay._step_count
  return {k: v.detach().cpu().numpy() for k, v in out.items()}


def restore(
  env: ManagerBasedRlEnv,
  snap: dict[str, torch.Tensor],
  rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
  """Write ``snap`` (``{name: [m, ...]}``, ``snapshot``'s keys, on the env's
  device) into envs ``rows [m]``; returns the observations to act on next
  (``{group: [m, ...]}``, the snapshot's, for the caller to place in the
  full batch)."""
  robot = _robot(env)
  robot.write_root_link_pose_to_sim(snap["root_pose"], env_ids=rows)
  robot.write_root_link_velocity_to_sim(snap["root_vel"], env_ids=rows)
  robot.write_joint_state_to_sim(snap["joint_pos"], snap["joint_vel"], env_ids=rows)
  am = env.action_manager
  am._action[rows] = snap["action"]
  am._prev_action[rows] = snap["prev_action"]
  am._prev_prev_action[rows] = snap["prev_prev_action"]
  for name, buf in _circular_buffers(env).items():
    if f"{name}.data" not in snap:
      continue
    if buf._buffer is None:
      raise RuntimeError(f"{name} is empty: step the env once before restoring.")
    data = snap[f"{name}.data"]  # [m, L, ...], oldest first.
    length = buf._max_len
    start = (buf._pointer + 1) % length
    for j in range(length):
      buf._buffer[(start + j) % length, rows] = data[:, j].to(buf._buffer.dtype)
    buf._num_pushes[rows] = snap[f"{name}.pushes"].long()
  for name, delay in _delay_buffers(env).items():
    delay._current_lags[rows] = snap[f"{name}.lags"].long()
    delay._step_count[rows] = snap[f"{name}.steps"].long()
  env.sim.forward()
  return {k[4:]: v for k, v in snap.items() if k.startswith("obs.")}
