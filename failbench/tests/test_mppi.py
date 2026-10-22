"""Tests for the MPPI planner: input parity with the data builder, the
obstacle sweep and the planner on a synthetic map."""

import torch
from failbench.data.build import _integrate
from failbench.mppi.costmap import FootprintCfg, distance_field, obstacle_cells, sweep
from failbench.mppi.inputs import HistoryBuffer, crop_center, integrate_plan
from failbench.mppi.models import ZeroDriftModel
from failbench.mppi.planner import Mppi, MppiCfg
from failbench.mppi.sampler import PlanSampler, SamplerCfg
from failbench.sensing import cell_centres

RES = 0.05


def test_integrate_plan_matches_the_builders_cmd_path():
  torch.manual_seed(0)
  plan = torch.rand(4, 50, 3) - 0.5
  # build.py integrates the 50 Hz command and reads every 5th step.
  ref = _integrate(plan.repeat_interleave(5, dim=1), 0.02)[:, 4::5]
  assert torch.allclose(integrate_plan(plan, 0.02, 5), ref, atol=1e-5)


def test_history_matches_the_builders_window():
  # build.py: history at step s is steps s - 19 r .. s (stride r), oldest first.
  hist = HistoryBuffer(1, 20, 1, "cpu")
  r, s = 5, 200
  for step in range(r, s + 1, r):
    hist.push(torch.tensor([[float(step)]]))
  expected = s - r * torch.arange(19, -1, -1)
  assert torch.equal(hist.data[0, :, 0], expected.float())


def test_centre_crop_shares_the_model_grid():
  big, small = cell_centres(160, RES, "cpu"), cell_centres(64, RES, "cpu")
  crop = crop_center(big.permute(2, 0, 1), 64).permute(1, 2, 0)
  assert torch.allclose(crop, small)


def test_stairs_are_not_walls_but_crates_and_poles_are():
  x = torch.arange(160) * RES
  stairs = (torch.clamp((x - 2.0) // 0.3 + 1, min=0) * 0.14).expand(160, 160)
  assert not obstacle_cells(stairs[None], 0.18).any()
  crate = torch.zeros(1, 160, 160)
  crate[:, 70:76, 70:76] = 0.2
  pole = torch.zeros(1, 160, 160)
  pole[:, 80, 80] = 1.0
  assert obstacle_cells(crate, 0.18).any()
  assert obstacle_cells(pole, 0.18)[0, 79:82, 79:82].any()


def test_sweep_charges_walls_but_not_open_ground():
  cfg = FootprintCfg()
  h = torch.zeros(1, 160, 160)
  h[:, 100:104, :] = 0.8  # A wall 1 m to the left (+y), along x.
  dist = distance_field(obstacle_cells(h, cfg.wall_jump_m), RES, cfg.max_dist_m)
  known = torch.ones_like(h, dtype=torch.bool)
  pose = torch.zeros(1, 2, 1, 3)
  pose[0, 1, 0, 1] = 0.9  # Second plan ends against the wall.
  std = torch.full_like(pose, 0.01)
  pen, unknown = sweep(dist, known, pose, std, RES, cfg)
  assert pen[0, 0].max() == 0 and pen[0, 1].max() > 0
  assert not unknown.any()
  # However uncertain, a plan in open ground isn't charged.
  free = distance_field(torch.zeros_like(h, dtype=torch.bool), RES, cfg.max_dist_m)
  pen, _ = sweep(free, known, pose[:, :1], torch.full_like(pose[:, :1], 5.0), RES, cfg)
  assert pen.max() == 0


def test_sampled_plans_respect_limits_and_committed_steps():
  sampler = PlanSampler(SamplerCfg(num_plans=32, compile=False), "cpu")
  warm = torch.rand(2, 50, 3)
  plans = sampler.sample(warm, warm[:, 0], torch.tensor([[5.0, 0.0]] * 2), None, 1)
  assert plans.shape == (2, 32, 50, 3)
  assert (plans[..., 0] >= 0).all() and (plans[..., 0] <= 1.2 + 1e-6).all()
  assert torch.equal(plans[:, :, 0], warm[:, None, 0].expand(2, 32, 3))


def _plan(h: torch.Tensor, goal: torch.Tensor, cycles: int = 10) -> torch.Tensor:
  torch.manual_seed(0)
  cfg = MppiCfg(sampler=SamplerCfg(num_plans=128, compile=False), delay_steps=0)
  planner = Mppi(cfg, ZeroDriftModel(), 1, "cpu")
  known = torch.ones_like(h, dtype=torch.bool)
  out = planner.step({}, h, known, RES, goal)
  for _ in range(cycles - 1):
    out = planner.step({}, h, known, RES, goal)
  return out["pred_path"][0]


def test_mppi_walks_toward_the_goal_in_the_open():
  goal = torch.tensor([[3.0, 0.5]])
  path = _plan(torch.zeros(1, 160, 160), goal)
  assert (path[-1, :2] - goal[0]).norm() < 1.0


def test_mppi_stops_short_of_a_wall_across_the_way():
  h = torch.zeros(1, 160, 160)
  h[:, :, 110:114] = 0.8  # A wall across the whole map 1.5 m ahead.
  path = _plan(h, torch.tensor([[3.0, 0.0]]))
  # Front circle (0.22 m ahead, radius 0.2) stays off the wall's face.
  assert path[:, 0].max() < 1.5 - 0.22 - 0.2
