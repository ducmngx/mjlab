"""Scripted mid-run joint faults and the post-fault contact log they produce.

Demo scenario, not a physical failure model: the nominal policy walks, then
at a sampled time each env's fault joint(s) fail in one of three ways and
the run continues *without a reset*, so what the robot hits on the way down
can be recorded (``heat_bench/scripts/fault_landing.py``).

- ``LIMIT``: ``jnt_range`` narrows to ``q_f +- limit_half_width_rad`` around
  the angle at fault time (clipped to the nominal range). The motor still
  drives the joint inside the band.
- ``LOCK``: ``jnt_range`` collapses to ``[q_f, q_f]`` -- a jammed gearbox
  holding the joint. The motor is left alone, so it keeps pushing (and
  heating) against the jam. MuJoCo limits are soft constraints, so the joint
  can give by a few mrad under impact.
- ``FREE``: the motor goes limp and the joint swings passively: through
  ``apply_actuator_health`` (``derate`` = 0) when the env has that term,
  otherwise by zeroing the joint's ``actuator_forcerange`` directly (envs
  without the heat_bench thermal stack, e.g. the Go2 HIM task).
- ``NONE``: a control trial. The fault "fires" (time and frame are recorded
  like any other) but nothing changes, so other stressors -- e.g. a payload
  -- can be measured on their own.

``jnt_range`` is writable per env at ``RecomputeLevel.none`` (same as
``dr.joint_limits``), so LIMIT/LOCK need no actuator-gain writes. Register
before ``actuator_health`` (``add_joint_fault_injection`` does) so a FREE
write lands in the same control step.

``fault_contact_sensor_cfg`` adds a whole-body robot-vs-terrain contact
sensor; ``contacts_in_base_frame`` turns its output into points and forces
on the robot, expressed in the trunk frame.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch

from heat_bench.envs_mjlab.actuator_health import (
  ActuatorState,
  apply_actuator_health,
  ctrl_ids_for_joints,
)
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.viewer.debug_visualizer import DebugVisualizer

FAULT_CONTACT_SENSOR = "fault_contacts"


class FaultMode(IntEnum):
  LIMIT = 0
  LOCK = 1
  FREE = 2
  NONE = 3


_MODE_STATE = {
  FaultMode.LIMIT: ActuatorState.LIMITED,
  FaultMode.LOCK: ActuatorState.LOCKED,
  FaultMode.FREE: ActuatorState.FREE,
  FaultMode.NONE: ActuatorState.HEALTHY,
}
_MODE_COLOR = {
  FaultMode.LIMIT: (1.0, 0.8, 0.0, 0.9),
  FaultMode.LOCK: (1.0, 0.1, 0.1, 0.9),
  FaultMode.FREE: (0.2, 0.5, 1.0, 0.9),
  FaultMode.NONE: (0.7, 0.7, 0.7, 0.6),
}


def parse_fault_modes(modes: tuple[str, ...]) -> tuple[FaultMode, ...]:
  try:
    return tuple(FaultMode[m.upper()] for m in modes)
  except KeyError as e:
    raise ValueError(
      f"Unknown fault mode {e}; expected any of {[m.name.lower() for m in FaultMode]}."
    ) from None


def fault_contact_sensor_cfg(num_slots: int = 2) -> ContactSensorCfg:
  """Every Go1 collision geom (trunk, head, hips, thighs, calves, feet)
  against the terrain, with global-frame position and force per contact."""
  return ContactSensorCfg(
    name=FAULT_CONTACT_SENSOR,
    primary=ContactMatch(mode="geom", pattern=r".*_collision.*", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force", "pos", "normal", "tangent"),
    reduce="none",
    num_slots=num_slots,
    global_frame=True,
  )


def contacts_in_base_frame(
  sensor: ContactSensor, asset: "Entity"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Current terrain contacts on the robot.

  Returns ``(pos_b, force_b, pos_w, force_w, valid)``: positions and forces
  in the trunk frame (origin at the root link) and in the world frame, each
  ``[B, N, 3]``, plus a ``[B, N]`` mask of filled slots. Contact ``n``
  belongs to geom ``sensor.primary_names[n // num_slots]``. Forces are the
  force the terrain exerts *on the robot*.
  """
  data = sensor.data
  assert data.pos is not None and data.force is not None and data.normal is not None
  # Unfilled slots are zeroed, including the unit normal.
  valid = data.normal.norm(dim=-1) > 1e-6
  # The sensor's force is the primary's push on the secondary along the
  # primary -> secondary normal; the robot feels the opposite.
  force_w = -data.force
  pos_w = data.pos
  root_pos = asset.data.root_link_pos_w[:, None, :]
  root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, pos_w.shape[1], -1)
  pos_b = quat_apply_inverse(root_quat, pos_w - root_pos)
  force_b = quat_apply_inverse(root_quat, force_w)
  return pos_b, force_b, pos_w, force_w, valid


class inject_joint_fault:
  """Fail sampled joints at a sampled time per episode; see module docstring.

  Use with ``mode="step"``, registered before ``actuator_health`` when there
  is one (``health_term``; ``None`` makes FREE write the force range itself).
  At each reset every env draws ``num_fault_joints`` distinct joints from
  ``joint_names``, one mode from ``modes``, and a fault time uniformly in
  ``fault_time_s``. The fault latches until the next reset.
  """

  model_fields = ("jnt_range", "actuator_forcerange")
  recompute = RecomputeLevel.none

  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    params = cfg.params
    asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
    self._env = env
    self._asset = env.scene[asset_cfg.name]
    self._health_term_name: str | None = params.get("health_term")
    self._health: apply_actuator_health | None = None

    local_ids, self.candidate_names = self._asset.find_joints(
      tuple(params["joint_names"]), preserve_order=True
    )
    self._candidate_local = torch.tensor(local_ids, device=env.device)
    self.modes = parse_fault_modes(tuple(params["modes"]))
    self._mode_ids = torch.tensor([int(m) for m in self.modes], device=env.device)
    self._mode_state = torch.tensor(
      [int(_MODE_STATE[m]) for m in FaultMode], device=env.device, dtype=torch.int8
    )
    """``ActuatorState`` for each ``FaultMode``, indexed by mode."""
    self.num_fault_joints: int = params.get("num_fault_joints", 1)
    if not 1 <= self.num_fault_joints <= len(local_ids):
      raise ValueError(
        f"num_fault_joints={self.num_fault_joints} must be in "
        f"[1, {len(local_ids)}] (the number of candidate joints)."
      )
    self._time_range: tuple[float, float] = tuple(params["fault_time_s"])
    self._half_width: float = params.get("limit_half_width_rad", 0.15)

    self.joint_names: list[str] = list(self._asset.joint_names)
    self._global_jnt_ids = self._asset.indexing.joint_ids.to(env.device)
    self._ctrl_ids = ctrl_ids_for_joints(
      self._asset, list(range(len(self.joint_names))), self.joint_names
    ).to(env.device)
    self._nominal_forcerange: torch.Tensor | None = None
    """Per-env ``actuator_forcerange`` of each joint's motor before any FREE
    fault, [B, J, 2]; read lazily, once fields are expanded per env. Only
    used without a health term."""
    self.nominal_jnt_range = env.sim.model.jnt_range[:, self._global_jnt_ids].clone()
    """Per-env ``jnt_range`` of the asset's joints before any fault, [B, J, 2]."""

    n, k = env.num_envs, self.num_fault_joints
    self.fault_joint = torch.zeros(n, k, dtype=torch.long, device=env.device)
    """Asset-local joint index of each env's fault joints, [B, K]."""
    self.fault_mode = torch.zeros(n, dtype=torch.long, device=env.device)
    """``FaultMode`` per env."""
    self.fault_time_s = torch.zeros(n, device=env.device)
    self.faulted = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.fault_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    """``episode_length_buf`` value when the fault fired; -1 before."""
    self.fault_qpos = torch.zeros(n, k, device=env.device)
    self.fault_base_pos_w = torch.zeros(n, 3, device=env.device)
    self.fault_base_quat_w = torch.zeros(n, 4, device=env.device)
    # Sample only: jnt_range isn't expanded per env until the EventManager
    # has built every term, and it still holds the nominal range anyway.
    self._sample(torch.arange(n, device=env.device))

  def _sample(self, rows: torch.Tensor) -> None:
    device = self._env.device
    n = rows.numel()
    picks = torch.rand(n, len(self._candidate_local), device=device).argsort(dim=-1)
    self.fault_joint[rows] = self._candidate_local[picks[:, : self.num_fault_joints]]
    self.fault_mode[rows] = self._mode_ids[
      torch.randint(len(self._mode_ids), (n,), device=device)
    ]
    lo, hi = self._time_range
    self.fault_time_s[rows] = lo + (hi - lo) * torch.rand(n, device=device)
    self.faulted[rows] = False
    self.fault_step[rows] = -1

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    rows = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
    self._sample(rows)
    self._env.sim.model.jnt_range[rows[:, None], self._global_jnt_ids] = (
      self.nominal_jnt_range[rows]
    )
    if self._nominal_forcerange is not None:
      self._env.sim.model.actuator_forcerange[rows[:, None], self._ctrl_ids] = (
        self._nominal_forcerange[rows]
      )

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    joint_names: tuple[str, ...],
    modes: tuple[str, ...],
    fault_time_s: tuple[float, float],
    health_term: str | None = None,
    num_fault_joints: int = 1,
    limit_half_width_rad: float = 0.15,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    del env_ids, joint_names, modes, fault_time_s, health_term
    del num_fault_joints, limit_half_width_rad, asset_cfg
    if self._health_term_name is None:
      if self._nominal_forcerange is None:
        self._nominal_forcerange = env.sim.model.actuator_forcerange[
          :, self._ctrl_ids
        ].clone()
      t = env.episode_length_buf.float() * env.step_dt
      fire = ~self.faulted & (t >= self.fault_time_s)
      if fire.any():
        self._fire(fire.nonzero().squeeze(-1))
      return
    health = self._health
    if health is None:
      health = env.event_manager.get_term_cfg(self._health_term_name).func
      # Fault joint indices are asset-local; the health buffers must match.
      if list(health.joint_names) != self.joint_names:
        raise ValueError("actuator_health must cover all joints in asset order.")
      self._health = health

    t = env.episode_length_buf.float() * env.step_dt
    fire = ~self.faulted & (t >= self.fault_time_s)
    if fire.any():
      self._fire(fire.nonzero().squeeze(-1))

    # The health term rewrites forcerange from derate every step and doesn't
    # latch derate, so re-assert FREE / the fault states every step.
    rows = self.faulted.nonzero().squeeze(-1)
    if rows.numel() == 0:
      return
    cols = self.fault_joint[rows]
    row_modes = self.fault_mode[rows]
    free = (row_modes == FaultMode.FREE)[:, None].expand_as(cols)
    health.derate[rows[:, None], cols] = torch.where(
      free, 0.0, health.derate[rows[:, None], cols]
    )
    state = self._mode_state[row_modes]
    health.state[rows[:, None], cols] = state[:, None].expand_as(cols)

  def _fire(self, rows: torch.Tensor) -> None:
    env = self._env
    cols = self.fault_joint[rows]
    q = self._asset.data.joint_pos[rows[:, None], cols]
    self.faulted[rows] = True
    self.fault_step[rows] = env.episode_length_buf[rows]
    self.fault_qpos[rows] = q
    self.fault_base_pos_w[rows] = self._asset.data.root_link_pos_w[rows]
    self.fault_base_quat_w[rows] = self._asset.data.root_link_quat_w[rows]

    nominal = self.nominal_jnt_range[rows[:, None], cols]
    half = torch.where(
      (self.fault_mode[rows] == FaultMode.LIMIT)[:, None], self._half_width, 0.0
    )
    new_lo = torch.maximum(q - half, nominal[..., 0])
    new_hi = torch.minimum(q + half, nominal[..., 1])
    new_range = torch.stack([new_lo, new_hi], dim=-1)
    # FREE and NONE leave the range alone.
    mode = self.fault_mode[rows]
    keep = ((mode == FaultMode.FREE) | (mode == FaultMode.NONE))[:, None, None]
    env.sim.model.jnt_range[rows[:, None], self._global_jnt_ids[cols]] = torch.where(
      keep, nominal, new_range
    )
    # Without a health term, FREE zeroes the motor's force range here; it
    # stays zero (latched) until the next reset restores it.
    if self._health_term_name is None:
      free = mode == FaultMode.FREE
      if free.any():
        free_rows = rows[free]
        ctrl = self._ctrl_ids[self.fault_joint[free_rows]]
        env.sim.model.actuator_forcerange[free_rows[:, None], ctrl] = 0.0

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    """Faulted joints as spheres colored by mode; terrain contacts as small
    spheres with arrows showing the force on the robot."""
    env = self._env
    env_ids = list(visualizer.get_env_indices(env.num_envs))
    if not env_ids:
      return
    idx = torch.tensor(env_ids, device=env.device)
    anchors = env.sim.data.xanchor[idx][:, self._global_jnt_ids].cpu().numpy()
    faulted = self.faulted[idx].cpu().numpy()
    modes = self.fault_mode[idx].cpu().numpy()
    joints = self.fault_joint[idx].cpu().numpy()
    sensor = env.scene.sensors.get(FAULT_CONTACT_SENSOR)
    contacts = None
    if isinstance(sensor, ContactSensor):
      _, _, pos_w, force_w, valid = contacts_in_base_frame(sensor, self._asset)
      contacts = (
        pos_w[idx].cpu().numpy(),
        force_w[idx].cpu().numpy(),
        valid[idx].cpu().numpy(),
      )
    for i in range(len(env_ids)):
      if faulted[i]:
        color = _MODE_COLOR[FaultMode(int(modes[i]))]
        for j in joints[i]:
          visualizer.add_sphere(anchors[i, j], 0.035, color)
      if contacts is None:
        continue
      pos, force, ok = contacts[0][i], contacts[1][i], contacts[2][i]
      for p, f in zip(pos[ok], force[ok], strict=True):
        visualizer.add_sphere(p, 0.012, (1.0, 0.3, 0.9, 0.9))
        # 1 cm per 10 N, capped so a hard impact stays on screen.
        norm = float(np.linalg.norm(f))
        length = min(norm * 1e-3, 0.4)
        if length > 1e-3:
          visualizer.add_arrow(p, p + f / norm * length, (1.0, 0.3, 0.9, 0.8), 0.008)
