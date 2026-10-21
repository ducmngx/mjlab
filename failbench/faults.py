"""Kinodynamic joint faults: a per-joint parameter vector and the model writes.

Every joint carries six parameters (``FAULT_PARAMS``), relative to nominal:

- ``torque``: scale on the motor's force range (0 = limp, as FREE).
- ``damping``: scale on ``dof_damping`` (gearbox drag; nominal 0.1 on Go2).
- ``friction``: scale on ``dof_frictionloss`` (nominal 0.2 N m on Go2).
- ``kp`` / ``kd``: scale on the position actuator's PD gains (driver fault).
- ``range``: half-width (rad) of the joint range around its fault-time
  angle, clipped to the nominal range. 0 = LOCK (jammed); ``RANGE_FREE``
  or more leaves the range alone.

``sample_fault_params`` draws a vector per env from a mixture of families
(``FaultFamily``); ``inject_kinodynamic_fault`` writes it at a sampled
fault time and restores the nominal model at reset. Like
``inject_joint_fault`` (heat_bench), whose LOCK/FREE/LIMIT modes are
special cases here, the run continues without a reset after the fault.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
from heat_bench.envs_mjlab.actuator_health import ctrl_ids_for_joints

from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg

FAULT_PARAMS = ("torque", "damping", "friction", "kp", "kd", "range")
TORQUE, DAMPING, FRICTION, KP, KD, RANGE = range(len(FAULT_PARAMS))
RANGE_FREE = math.pi
"""A ``range`` half-width at or above this leaves the joint range alone."""
NOMINAL = (1.0, 1.0, 1.0, 1.0, 1.0, RANGE_FREE)

FAULT_TERM = "kinodynamic_fault"


class FaultFamily(IntEnum):
  NOMINAL = 0
  """No change: the fault time is recorded, nothing is written."""
  SEVERE = 1
  """One joint: locked, limp (torque 0) or a torque limit of at most 0.3."""
  CONTINUOUS = 2
  """One joint, each parameter perturbed with probability 0.5."""
  MULTI_MILD = 3
  """Two to four joints, every parameter mildly perturbed."""


@dataclass(frozen=True)
class FaultMixture:
  """Family probabilities and parameter ranges for ``sample_fault_params``."""

  probs: tuple[float, float, float, float] = (0.15, 0.35, 0.30, 0.20)
  """Probability of each ``FaultFamily``, in enum order."""
  severe_torque_max: float = 0.3
  torque: tuple[float, float] = (0.2, 1.0)
  damping: tuple[float, float] = (1.0, 30.0)
  """Log-uniform."""
  friction: tuple[float, float] = (1.0, 30.0)
  """Log-uniform."""
  gain: tuple[float, float] = (0.2, 1.2)
  """Kp and Kd scale."""
  range_prob: float = 0.25
  """CONTINUOUS: probability the joint range also narrows."""
  range_half_width: tuple[float, float] = (0.05, 0.5)
  mild_joints: tuple[int, int] = (2, 4)
  mild_torque: tuple[float, float] = (0.5, 1.0)
  mild_damping: tuple[float, float] = (1.0, 10.0)
  mild_friction: tuple[float, float] = (1.0, 10.0)
  mild_gain: tuple[float, float] = (0.6, 1.1)


def _uniform(shape, lo: float, hi: float, device) -> torch.Tensor:
  return lo + (hi - lo) * torch.rand(shape, device=device)


def _log_uniform(shape, lo: float, hi: float, device) -> torch.Tensor:
  return torch.exp(_uniform(shape, math.log(lo), math.log(hi), device))


def sample_fault_params(
  n: int, num_joints: int, mix: FaultMixture, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Fault parameters ``[n, num_joints, 6]`` and family ``[n]`` per env."""
  params = torch.tensor(NOMINAL, device=device).repeat(n, num_joints, 1)
  probs = torch.tensor(mix.probs, device=device)
  family = torch.multinomial(probs / probs.sum(), n, replacement=True)
  rows = torch.arange(n, device=device)
  # One random joint per env (SEVERE, CONTINUOUS), or a random subset (MILD).
  order = torch.rand(n, num_joints, device=device).argsort(dim=-1)
  joint = order[:, 0]

  sev = rows[family == FaultFamily.SEVERE]
  if sev.numel():
    kind = torch.randint(3, (sev.numel(),), device=device)  # lock, limp, weak
    j = joint[sev]
    params[sev[kind == 0], j[kind == 0], RANGE] = 0.0
    params[sev[kind == 1], j[kind == 1], TORQUE] = 0.0
    weak = kind == 2
    params[sev[weak], j[weak], TORQUE] = _uniform(
      int(weak.sum()), 0.0, mix.severe_torque_max, device
    )

  con = rows[family == FaultFamily.CONTINUOUS]
  if con.numel():
    j, m = joint[con], con.numel()
    draws = (
      (TORQUE, _uniform(m, *mix.torque, device)),
      (DAMPING, _log_uniform(m, *mix.damping, device)),
      (FRICTION, _log_uniform(m, *mix.friction, device)),
      (KP, _uniform(m, *mix.gain, device)),
      (KD, _uniform(m, *mix.gain, device)),
    )
    hit = torch.rand(m, len(draws), device=device) < 0.5
    # At least one parameter changes, so the family never collapses to NOMINAL.
    none = ~hit.any(dim=-1)
    hit[none, torch.randint(len(draws), (int(none.sum()),), device=device)] = True
    for col, (p, value) in enumerate(draws):
      params[con, j, p] = torch.where(hit[:, col], value, params[con, j, p])
    narrow = torch.rand(m, device=device) < mix.range_prob
    params[con[narrow], j[narrow], RANGE] = _uniform(
      int(narrow.sum()), *mix.range_half_width, device
    )

  mild = rows[family == FaultFamily.MULTI_MILD]
  if mild.numel():
    lo, hi = mix.mild_joints
    k = torch.randint(lo, hi + 1, (mild.numel(),), device=device)
    rank = torch.empty_like(order[mild])
    rank.scatter_(
      1, order[mild], torch.arange(num_joints, device=device).expand_as(rank)
    )
    chosen = rank < k[:, None]
    shape = (mild.numel(), num_joints)
    draws = (
      (TORQUE, _uniform(shape, *mix.mild_torque, device)),
      (DAMPING, _log_uniform(shape, *mix.mild_damping, device)),
      (FRICTION, _log_uniform(shape, *mix.mild_friction, device)),
      (KP, _uniform(shape, *mix.mild_gain, device)),
      (KD, _uniform(shape, *mix.mild_gain, device)),
    )
    for p, value in draws:
      params[mild, :, p] = torch.where(chosen, value, params[mild, :, p])
  return params, family


def encode_fault(params: torch.Tensor) -> torch.Tensor:
  """Network encoding of ``[..., J, 6]`` parameters, flattened to
  ``[..., J * 6]``: torque, Kp, Kd as is; damping and friction as
  ``log(scale) / log(30)``; range as ``min(half_width, pi) / pi``. A
  healthy joint is (1, 0, 0, 1, 1, 1)."""
  out = params.clone()
  log30 = math.log(30.0)
  out[..., DAMPING] = params[..., DAMPING].clamp(min=1e-6).log() / log30
  out[..., FRICTION] = params[..., FRICTION].clamp(min=1e-6).log() / log30
  out[..., RANGE] = params[..., RANGE].clamp(max=RANGE_FREE) / RANGE_FREE
  return out.flatten(-2)


class inject_kinodynamic_fault:
  """Write each env's sampled fault parameters at its fault time.

  ``mode="step"`` event. At every reset each env draws a parameter vector
  (``sample_fault_params``) and a fault time uniformly in ``fault_time_s``,
  and the model is restored to nominal. Not for envs with heat_bench's
  ``actuator_health`` term, which rewrites the force range every step.
  """

  model_fields = (
    "jnt_range",
    "actuator_forcerange",
    "actuator_gainprm",
    "actuator_biasprm",
    "dof_damping",
    "dof_frictionloss",
  )
  recompute = RecomputeLevel.none

  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    params = cfg.params
    asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
    self._env = env
    self._asset = env.scene[asset_cfg.name]
    self.mixture: FaultMixture = params.get("mixture", FaultMixture())
    self._time_range: tuple[float, float] = tuple(params["fault_time_s"])

    self.joint_names: list[str] = list(self._asset.joint_names)
    num_j = len(self.joint_names)
    idx = self._asset.indexing
    self._jnt = idx.joint_ids.to(env.device)
    self._dof = idx.joint_v_adr.to(env.device)
    self._ctrl = ctrl_ids_for_joints(
      self._asset, list(range(num_j)), self.joint_names
    ).to(env.device)

    n = env.num_envs
    self.params = torch.zeros(n, num_j, len(FAULT_PARAMS), device=env.device)
    """Fault parameters per env and joint (``FAULT_PARAMS`` order)."""
    self.family = torch.zeros(n, dtype=torch.long, device=env.device)
    self.fault_time_s = torch.zeros(n, device=env.device)
    self.faulted = torch.zeros(n, dtype=torch.bool, device=env.device)
    self.fault_step = torch.full((n,), -1, dtype=torch.long, device=env.device)
    """``episode_length_buf`` value when the fault fired; -1 before."""
    self.fault_qpos = torch.zeros(n, num_j, device=env.device)
    self.fault_base_pos_w = torch.zeros(n, 3, device=env.device)
    self.fault_base_quat_w = torch.zeros(n, 4, device=env.device)
    self._sample(torch.arange(n, device=env.device))

  def _default(self, field: str, ids: torch.Tensor) -> torch.Tensor:
    return self._env.sim.get_default_field(field)[ids].to(self._env.device)

  def _sample(self, rows: torch.Tensor) -> None:
    p, fam = sample_fault_params(
      rows.numel(), len(self.joint_names), self.mixture, self._env.device
    )
    self.params[rows] = p
    self.family[rows] = fam
    lo, hi = self._time_range
    self.fault_time_s[rows] = _uniform(rows.numel(), lo, hi, self._env.device)
    self.faulted[rows] = False
    self.fault_step[rows] = -1

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    rows = torch.arange(self._env.num_envs, device=self._env.device)[env_ids]
    self._sample(rows)
    m, r = self._env.sim.model, rows[:, None]
    m.jnt_range[r, self._jnt] = self._default("jnt_range", self._jnt)
    m.actuator_forcerange[r, self._ctrl] = self._default(
      "actuator_forcerange", self._ctrl
    )
    m.actuator_gainprm[r, self._ctrl] = self._default("actuator_gainprm", self._ctrl)
    m.actuator_biasprm[r, self._ctrl] = self._default("actuator_biasprm", self._ctrl)
    m.dof_damping[r, self._dof] = self._default("dof_damping", self._dof)
    m.dof_frictionloss[r, self._dof] = self._default("dof_frictionloss", self._dof)

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    fault_time_s: tuple[float, float],
    mixture: FaultMixture | None = None,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    del env_ids, fault_time_s, mixture, asset_cfg
    t = env.episode_length_buf.float() * env.step_dt
    fire = ~self.faulted & (t >= self.fault_time_s)
    if fire.any():
      self._fire(fire.nonzero().squeeze(-1))

  def _fire(self, rows: torch.Tensor) -> None:
    env, data = self._env, self._asset.data
    m, r = env.sim.model, rows[:, None]
    p = self.params[rows]
    q = data.joint_pos[rows]
    self.faulted[rows] = True
    self.fault_step[rows] = env.episode_length_buf[rows]
    self.fault_qpos[rows] = q
    self.fault_base_pos_w[rows] = data.root_link_pos_w[rows]
    self.fault_base_quat_w[rows] = data.root_link_quat_w[rows]

    nominal = self._default("jnt_range", self._jnt).expand(len(rows), -1, -1)
    half = p[..., RANGE]
    narrowed = torch.stack(
      [
        torch.maximum(q - half, nominal[..., 0]),
        torch.minimum(q + half, nominal[..., 1]),
      ],
      dim=-1,
    )
    keep = (half >= RANGE_FREE)[..., None]
    m.jnt_range[r, self._jnt] = torch.where(keep, nominal, narrowed)

    force = self._default("actuator_forcerange", self._ctrl)
    m.actuator_forcerange[r, self._ctrl] = force * p[..., TORQUE, None]
    gain = self._default("actuator_gainprm", self._ctrl)[:, 0]
    bias = self._default("actuator_biasprm", self._ctrl)
    m.actuator_gainprm[r, self._ctrl, 0] = gain * p[..., KP]
    m.actuator_biasprm[r, self._ctrl, 1] = bias[:, 1] * p[..., KP]
    m.actuator_biasprm[r, self._ctrl, 2] = bias[:, 2] * p[..., KD]
    damping = self._default("dof_damping", self._dof)
    m.dof_damping[r, self._dof] = damping * p[..., DAMPING]
    friction = self._default("dof_frictionloss", self._dof)
    m.dof_frictionloss[r, self._dof] = friction * p[..., FRICTION]
