"""A rigid box payload strapped to the trunk, with explicit mass and mount.

Unlike the yaml ``payload`` section (``dr.pseudo_inertia``'s ``alpha``,
which makes the trunk uniformly denser and leaves its COM in place), this
models a load sitting *on* the robot: a solid box of ``mass_kg`` whose
center is at ``offset`` in the trunk frame. Mounted on top, it raises the
COM, which is what changes how and where a robot falls.

The trunk and box are combined exactly by adding their pseudo-inertia
matrices about the trunk origin (pseudo-inertia is additive over rigidly
attached bodies), reusing the helpers behind ``dr.pseudo_inertia``. The
result is always built from the model defaults, so resampling at every
reset never accumulates. It replaces any startup COM randomization on the
trunk (``base_com``), like ``dr.pseudo_inertia`` does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import _select_default_values
from mjlab.envs.mdp.dr.body import (
  _decompose_pseudo_inertia_J,
  _reconstruct_pseudo_inertia_J,
)
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.viewer.debug_visualizer import DebugVisualizer

# Go1 trunk_collision is a box of half-extents (0.13, 0.09, 0.05) m (go1.xml);
# a box of these half-extents resting on it has its center at z = 0.09.
DEFAULT_HALF_EXTENTS = (0.12, 0.08, 0.04)
DEFAULT_OFFSET_RANGE = ((-0.05, 0.05), (-0.03, 0.03), (0.09, 0.09))


def box_principal_inertia(mass: torch.Tensor, half_extents: torch.Tensor):
  """Principal moments of a solid box, ``[..., 3]``."""
  a2, b2, c2 = (half_extents**2).unbind(-1)
  return (mass / 3.0).unsqueeze(-1) * torch.stack([b2 + c2, a2 + c2, a2 + b2], dim=-1)


class attach_payload:
  """Sample a box payload per env and fold it into the trunk's inertials.

  Use with ``mode="reset"`` (new payload every episode) or ``"startup"``.
  ``mass_kg`` and ``offset`` hold each env's sample, for logging.
  """

  model_fields = (
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "body_subtreemass",
    "dof_invweight0",
    "body_invweight0",
    "tendon_length0",
    "tendon_invweight0",
    "actuator_acc0",
  )
  recompute = RecomputeLevel.set_const

  def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedRlEnv"):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._env = env
    asset = env.scene[asset_cfg.name]
    self._asset = asset
    body_ids, _ = asset.find_bodies(asset_cfg.body_names)
    if len(body_ids) != 1:
      raise ValueError("attach_payload needs exactly one body (the trunk).")
    self._local_body = body_ids[0]
    self._body_id = asset.indexing.body_ids[body_ids].to(env.device)
    self.mass_kg = torch.zeros(env.num_envs, device=env.device)
    """Payload mass per env (kg)."""
    self.offset = torch.zeros(env.num_envs, 3, device=env.device)
    """Payload box center per env, trunk frame (m)."""

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_kg_range: tuple[float, float],
    offset_range: tuple[tuple[float, float], ...] = DEFAULT_OFFSET_RANGE,
    half_extents: tuple[float, float, float] = DEFAULT_HALF_EXTENTS,
  ) -> None:
    del asset_cfg
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    n = env_ids.numel()
    device = env.device

    lo, hi = mass_kg_range
    mass = lo + (hi - lo) * torch.rand(n, device=device)
    bounds = torch.tensor(offset_range, device=device)
    offset = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(
      n, 3, device=device
    )
    self.mass_kg[env_ids] = mass
    self.offset[env_ids] = offset

    def default(field: str) -> torch.Tensor:
      return _select_default_values(env, field, env_ids, self._body_id)[:, 0]

    j_trunk = _reconstruct_pseudo_inertia_J(
      default("body_mass"),
      default("body_ipos"),
      default("body_inertia"),
      default("body_iquat"),
    )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(n, 4)
    j_box = _reconstruct_pseudo_inertia_J(
      mass,
      offset,
      box_principal_inertia(mass, torch.tensor(half_extents, device=device)),
      identity,
    )
    new_mass, ipos, inertia, iquat = _decompose_pseudo_inertia_J(j_trunk + j_box)

    model = env.sim.model
    body = self._body_id.expand(n)
    model.body_mass[env_ids, body] = new_mass
    model.body_ipos[env_ids, body] = ipos
    model.body_inertia[env_ids, body] = inertia
    model.body_iquat[env_ids, body] = iquat

  def debug_vis(self, visualizer: "DebugVisualizer") -> None:
    """Payload center as a grey sphere, scaled with its mass."""
    data = self._asset.data
    for env_idx in visualizer.get_env_indices(self._env.num_envs):
      mass = float(self.mass_kg[env_idx])
      if mass <= 0.0:
        continue
      pos = data.body_link_pos_w[env_idx, self._local_body]
      quat = data.body_link_quat_w[env_idx, self._local_body]
      rot = matrix_from_quat(quat)
      center = (pos + rot @ self.offset[env_idx]).cpu().numpy()
      visualizer.add_sphere(center, 0.03 + 0.005 * mass, (0.5, 0.5, 0.5, 0.8))
