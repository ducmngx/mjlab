"""Gearbox load observation: the torque the transmission actually carries.

``GearLoadObservation`` is a read-only observation term. Each physics
substep it computes, per joint::

  tau_gear = qfrc_actuator - armature * qacc

``armature`` is the motor rotor's reflected inertia (mjlab's Go1 sets it to
``ROTOR_INERTIA * N**2``, ``go1_constants.py``), and the rotor sits on the
input side of the gearbox. When the joint accelerates, the gear teeth must
also accelerate the rotor, so the load through them is the motor's torque
minus what went into the rotor. At a footfall the joint is stopped within a
few milliseconds, and braking the rotor puts a torque spike through the
gears on top of the motor torque -- the impact-transmission effect of high
reflected inertia (Wensing et al. 2017, heat_bench/README.md References
[9]). ``qfrc_actuator`` itself is clamped at the effort limit, so this term
is how a policy's impacts show up; measured with a rough-terrain Go1 policy,
knee gear load reaches ~2x the effort limit in normal walking and 3-4x under
pushes.

Purely observational: it doesn't change torque, and it isn't fed to the
thermal model -- motor current follows the motor's torque (what
``ThermalEnergyObservation`` already uses), not the rotor-braking share
carried by the gears. Absolute peaks depend on the contact model and
physics step (and the sim's gears are rigid), so compare policies within
the sim rather than reading them as hardware loads.

Impacts last a few milliseconds, so like ``ThermalEnergyObservation`` this
term wraps ``Simulation.step`` to see every physics substep, keeping the
per-joint peak |tau_gear| over each control step. It wraps whatever
``env.sim.step`` is when it's built (normally the thermal term's wrapper),
so the hooks chain and each still runs exactly once per substep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from heat_bench.envs_mjlab.actuator_health import ctrl_ids_for_joints
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.observation_manager import ObservationTermCfg


class GearLoadObservation:
  """Per-joint peak gearbox load over each control step."""

  def __init__(self, cfg: "ObservationTermCfg", env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._env = env
    self._asset = env.scene[asset_cfg.name]
    joint_ids, joint_names = self._asset.find_joints(asset_cfg.joint_names)
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self.joint_names: list[str] = joint_names
    """Resolved joint names, in the same order as ThermalEnergyObservation's
    (both resolve ``(".*",)`` with ``find_joints``)."""
    self._dof_ids = self._asset.indexing.joint_v_adr[self._joint_ids]

    # Snapshot the spec effort limit now: apply_actuator_health rewrites
    # actuator_forcerange every step (thermal derate, voltage limit, dead
    # joints), and the ratio should stay relative to the undegraded limit.
    ctrl_ids = ctrl_ids_for_joints(self._asset, joint_ids, joint_names).to(env.device)
    self.effort_limit = env.sim.model.actuator_forcerange[:, ctrl_ids, 1].clone()
    """Per-joint spec effort limit (N·m), shape [1 or num_envs, num_joints]."""

    num_joints = len(joint_ids)
    self.last_peak_gear_load = torch.zeros(env.num_envs, num_joints, device=env.device)
    """Peak |tau_gear| (N·m) over the last control step's substeps."""
    self.last_peak_gear_load_ratio = torch.zeros_like(self.last_peak_gear_load)
    """``last_peak_gear_load`` divided by the spec effort limit."""
    self._peak_accum = torch.zeros_like(self.last_peak_gear_load)
    self._substep_count = 0

    self._orig_sim_step = env.sim.step
    env.sim.step = self._accumulate_substep  # ty: ignore[invalid-assignment]

  def gear_load(self) -> torch.Tensor:
    """Current tau_gear (N·m) per joint, signed."""
    tau = self._asset.data.qfrc_actuator[:, self._joint_ids]
    qacc = self._asset.data.joint_acc[:, self._joint_ids]
    # Read live: dr.dof_armature may rewrite it at reset.
    armature = self._env.sim.model.dof_armature[:, self._dof_ids]
    return tau - armature * qacc

  def _accumulate_substep(self) -> None:
    self._orig_sim_step()
    torch.maximum(self._peak_accum, self.gear_load().abs(), out=self._peak_accum)
    self._substep_count += 1

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    # Deliberately keeps the accumulator: auto-reset runs after the
    # decimation loop but before this step's observation compute, so the
    # rows being reset still hold the episode's final control step -- often
    # the impact that ended it. __call__ publishes and drains them right
    # after, so nothing leaks into the new episode.
    del env_ids

  def step_peak_ratio(self) -> torch.Tensor:
    """This control step's per-joint peak ratio, readable mid-step.

    The metrics manager computes after the decimation loop but before
    observations, so ``last_peak_gear_load_ratio`` would still hold the
    previous step there; this reads the live accumulator instead."""
    if self._substep_count == 0:
      return self.last_peak_gear_load_ratio
    return self._peak_accum / self.effort_limit

  def __call__(
    self, env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg, **kwargs
  ) -> torch.Tensor:
    del env, asset_cfg, kwargs
    if self._substep_count > 0:
      peak = self._peak_accum.clone()
    else:
      # No physics substep yet (observation compute during env.reset()).
      peak = self.gear_load().abs()
    self._peak_accum.zero_()
    self._substep_count = 0

    self.last_peak_gear_load = peak
    self.last_peak_gear_load_ratio = peak / self.effort_limit
    return self.last_peak_gear_load_ratio


def _gear_load_term(env: "ManagerBasedRlEnv", obs_group: str, obs_term: str):
  return env.observation_manager.get_term_cfg(obs_group, obs_term).func


def gear_load_peak_ratio(
  env: "ManagerBasedRlEnv", obs_group: str, obs_term: str
) -> torch.Tensor:
  """Worst joint's peak gear load / effort limit this step. Reduce with
  ``max`` for the episode's harshest impact, ``mean`` for typical load."""
  term = _gear_load_term(env, obs_group, obs_term)
  return term.step_peak_ratio().max(dim=-1).values
