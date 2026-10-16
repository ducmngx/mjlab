"""Tests for the explicit box payload (attach_payload)."""

import os
from unittest.mock import Mock

import torch
from heat_bench.envs_mjlab.payload import attach_payload, box_principal_inertia

from mjlab.asset_zoo.robots import get_go1_robot_cfg
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import matrix_from_quat


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def _make_term(device: str, num_envs: int):
  entity = Entity(get_go1_robot_cfg())
  model = entity.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=device)
  sim.expand_model_fields(attach_payload.model_fields)
  entity.initialize(model, sim.model, sim.data, device)
  env = Mock()
  env.num_envs = num_envs
  env.device = device
  env.scene = {"robot": entity}
  env.sim = sim
  asset_cfg = SceneEntityCfg("robot", body_names=("trunk",))
  term = attach_payload(
    EventTermCfg(func=attach_payload, mode="reset", params={"asset_cfg": asset_cfg}),
    env,
  )
  return term, env, asset_cfg


def test_payload_adds_mass_and_moves_com_by_weighted_average():
  device = get_test_device()
  term, env, asset_cfg = _make_term(device, num_envs=2)
  model = env.sim.model
  body = int(term._body_id[0])
  m0 = model.body_mass[0, body].clone()
  c0 = model.body_ipos[0, body].clone()
  offset = ((0.02, 0.02), (0.0, 0.0), (0.09, 0.09))

  # Twice, to show resampling rebuilds from defaults instead of stacking.
  for _ in range(2):
    term(env, None, mass_kg_range=(5.0, 5.0), offset_range=offset, asset_cfg=asset_cfg)
  mass = model.body_mass[:, body]
  torch.testing.assert_close(mass, (m0 + 5.0).expand(2))
  expected_com = (m0 * c0 + 5.0 * torch.tensor([0.02, 0.0, 0.09], device=device)) / (
    m0 + 5.0
  )
  torch.testing.assert_close(model.body_ipos[:, body], expected_com.expand(2, 3))
  assert (model.body_inertia[:, body] > 0).all()
  torch.testing.assert_close(term.mass_kg, torch.full((2,), 5.0, device=device))


def test_payload_inertia_matches_parallel_axis_theorem():
  device = get_test_device()
  term, env, asset_cfg = _make_term(device, num_envs=1)
  model = env.sim.model
  body = int(term._body_id[0])

  def com_inertia(env_idx: int) -> torch.Tensor:
    rot = matrix_from_quat(model.body_iquat[env_idx, body])
    return rot @ torch.diag(model.body_inertia[env_idx, body]) @ rot.T

  def shift(inertia, mass, d):
    return inertia + mass * (d.dot(d) * torch.eye(3, device=device) - torch.outer(d, d))

  # Clone: plain indexing returns views that the payload write would change.
  m0 = model.body_mass[0, body].clone()
  c0 = model.body_ipos[0, body].clone()
  i0 = com_inertia(0)
  p = torch.tensor([0.0, 0.0, 0.09], device=device)
  half = torch.tensor([0.12, 0.08, 0.04], device=device)
  m = torch.tensor(5.0, device=device)
  term(
    env,
    None,
    mass_kg_range=(5.0, 5.0),
    offset_range=((0.0, 0.0), (0.0, 0.0), (0.09, 0.09)),
    half_extents=(0.12, 0.08, 0.04),
    asset_cfg=asset_cfg,
  )
  com = (m0 * c0 + m * p) / (m0 + m)
  expected = shift(i0, m0, c0 - com) + shift(
    torch.diag(box_principal_inertia(m, half)), m, p - com
  )
  torch.testing.assert_close(com_inertia(0), expected, rtol=1e-4, atol=1e-6)
