"""Single-joint actuator faults: a per-joint parameter vector and the model
writes.

Every joint carries two parameters (``FAULT_PARAMS``):

- ``torque``: scale on the motor's force range (0 = limp, as heat_bench's
  FREE; between, a motor or driver losing power).
- ``lock``: 1 = the joint jams at its fault-time angle (heat_bench's LOCK,
  a seized gearbox), 0 = free to move.

``sample_fault_params`` draws a vector per env from a mixture of families
(``FaultFamily``), each faulting at most one joint;
``inject_kinodynamic_fault`` writes it when the fault fires (at a sampled
time, or on stair tiles where the robot reaches a sampled distance from the
tile centre) and restores the nominal model at reset. Like ``inject_joint_fault`` (heat_bench), the
run continues without a reset after the fault.
"""

from __future__ import annotations

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

FAULT_PARAMS = ("torque", "lock")
TORQUE, LOCK = range(len(FAULT_PARAMS))
NOMINAL = (1.0, 0.0)

FAULT_TERM = "kinodynamic_fault"


class FaultFamily(IntEnum):
  NOMINAL = 0
  """No change: the fault time is recorded, nothing is written."""
  LOCK = 1
  """One joint jammed at its fault-time angle."""
  WEAK = 2
  """One joint's torque limit scaled into ``weak_torque`` (or to 0, limp)."""


@dataclass(frozen=True)
class FaultMixture:
  """Family probabilities and parameter ranges for ``sample_fault_params``."""

  probs: tuple[float, float, float] = (0.15, 0.40, 0.45)
  """Probability of each ``FaultFamily``, in enum order."""
  weak_torque: tuple[float, float] = (0.0, 0.5)
  """Uniform torque scale for WEAK."""
  limp_prob: float = 0.25
  """WEAK: probability the torque is exactly 0 (fully limp)."""


def _uniform(shape, lo: float, hi: float, device) -> torch.Tensor:
  return lo + (hi - lo) * torch.rand(shape, device=device)


def sample_fault_params(
  n: int, num_joints: int, mix: FaultMixture, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
  """Fault parameters ``[n, num_joints, 2]`` and family ``[n]`` per env."""
  params = torch.tensor(NOMINAL, device=device).repeat(n, num_joints, 1)
  probs = torch.tensor(mix.probs, device=device)
  family = torch.multinomial(probs / probs.sum(), n, replacement=True)
  joint = torch.randint(num_joints, (n,), device=device)

  lock = (family == FaultFamily.LOCK).nonzero().squeeze(-1)
  params[lock, joint[lock], LOCK] = 1.0

  weak = (family == FaultFamily.WEAK).nonzero().squeeze(-1)
  m = weak.numel()
  torque = _uniform(m, *mix.weak_torque, device)
  limp = torch.rand(m, device=device) < mix.limp_prob
  params[weak, joint[weak], TORQUE] = torch.where(limp, 0.0, torque)
  return params, family


def faulted_joint(params: torch.Tensor) -> torch.Tensor:
  """Index of the faulted joint per env (``[..., J, 2]`` -> ``[...]``), or
  -1 for a healthy robot."""
  touched = (params[..., TORQUE] < 1.0) | (params[..., LOCK] > 0.5)
  return torch.where(touched.any(-1), touched.float().argmax(-1), -1)


def encode_fault(params: torch.Tensor) -> torch.Tensor:
  """Network encoding of ``[..., J, 2]`` parameters: the parameters as is,
  flattened to ``[..., J * 2]``. A healthy joint is (1, 0)."""
  return params.flatten(-2)


class inject_kinodynamic_fault:
  """Write each env's sampled fault parameters when its fault fires.

  ``mode="step"`` event. At every reset each env draws a parameter vector
  (``sample_fault_params``) and a fault time uniformly in ``fault_time_s``,
  and the model is restored to nominal. With ``fire_radius`` the fault
  instead fires once the trunk is a sampled distance in ``fire_radius``
  (Chebyshev, m) from its terrain origin, the centre of a stair pyramid, and
  at least ``min_time_s`` in; ``fault_time_s`` is then the timeout. Not for envs with heat_bench's
  ``actuator_health`` term, which rewrites the force range every step.
  """

  model_fields = ("jnt_range", "actuator_forcerange")
  recompute = RecomputeLevel.none

  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    params = cfg.params
    asset_cfg: SceneEntityCfg = params.get("asset_cfg", SceneEntityCfg("robot"))
    self._env = env
    self._asset = env.scene[asset_cfg.name]
    self.mixture: FaultMixture = params.get("mixture", FaultMixture())
    self._time_range: tuple[float, float] = tuple(params["fault_time_s"])
    radius = params.get("fire_radius")
    self._radius_range: tuple[float, float] | None = (
      None if radius is None else tuple(radius)
    )
    self._min_time_s: float = params.get("min_time_s", 0.0)

    self.joint_names: list[str] = list(self._asset.joint_names)
    num_j = len(self.joint_names)
    idx = self._asset.indexing
    self._jnt = idx.joint_ids.to(env.device)
    self._ctrl = ctrl_ids_for_joints(
      self._asset, list(range(num_j)), self.joint_names
    ).to(env.device)

    n = env.num_envs
    self.params = torch.zeros(n, num_j, len(FAULT_PARAMS), device=env.device)
    """Fault parameters per env and joint (``FAULT_PARAMS`` order)."""
    self.family = torch.zeros(n, dtype=torch.long, device=env.device)
    self.fault_time_s = torch.zeros(n, device=env.device)
    """Fault time, or the timeout with ``fire_radius``."""
    self.fire_radius = torch.full((n,), float("inf"), device=env.device)
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
    if self._radius_range is not None:
      self.fire_radius[rows] = _uniform(
        rows.numel(), *self._radius_range, self._env.device
      )
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

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    fault_time_s: tuple[float, float],
    mixture: FaultMixture | None = None,
    fire_radius: tuple[float, float] | None = None,
    min_time_s: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    del env_ids, fault_time_s, mixture, fire_radius, min_time_s, asset_cfg
    t = env.episode_length_buf.float() * env.step_dt
    due = t >= self.fault_time_s
    if self._radius_range is not None:
      rel = self._asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
      out = rel.abs().max(dim=-1).values >= self.fire_radius
      due |= out & (t >= self._min_time_s)
    fire = ~self.faulted & due
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
    locked = (p[..., LOCK] > 0.5)[..., None]
    m.jnt_range[r, self._jnt] = torch.where(locked, q[..., None], nominal)
    force = self._default("actuator_forcerange", self._ctrl)
    m.actuator_forcerange[r, self._ctrl] = force * p[..., TORQUE, None]
