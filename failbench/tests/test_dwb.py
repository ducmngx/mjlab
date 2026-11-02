"""Tests for the Nav2 DWB port: its velocity window and rollouts, and closed
loop on small worlds of boxes, the robot moved by the shared motion model."""

import torch
from failbench.mppi.dwb import Dwb, DwbCfg, rollout, velocity_grid
from failbench.mppi.nav2 import nav2_costmap
from failbench.tests.test_nav2 import RES, _drive, _local_heights


def test_the_grid_is_the_dynamic_window_around_the_current_speed():
  cfg = DwbCfg()
  v0 = torch.tensor([[0.6, 0.0, 0.0]])
  g = velocity_grid(v0, cfg)
  assert g.shape == (1, 20 * 1 * 20, 3)
  # One control period of acceleration either side, inside the limits.
  assert torch.isclose(g[0, :, 0].max(), torch.tensor(0.6 + 2.5 * 0.1))
  assert torch.isclose(g[0, :, 0].min(), torch.tensor(0.6 - 2.5 * 0.1))
  # Nav2's stock DWB is a diff-drive's: it never sidesteps (its critics
  # can't tell a crab from walking straight).
  assert (g[0, :, 1] == 0).all()
  stop = velocity_grid(torch.zeros(1, 3), cfg)
  assert (stop[0, :, 0] >= 0).all()


def test_a_rollout_ramps_to_its_twist_within_the_acceleration_limit():
  cfg = DwbCfg()
  traj, v = rollout(torch.tensor([[[1.2, 0.0, 0.0]]]), torch.zeros(1, 3), cfg)
  assert v.shape[2] == round(1.7 / 0.1)
  assert (v[0, 0, 1:, 0].diff() <= 2.5 * 0.1 + 1e-6).all()
  assert torch.isclose(v[0, 0, -1, 0], torch.tensor(1.2))
  assert traj[0, 0, -1, 0] > 1.0 and traj[0, 0, -1, 1].abs() < 1e-6


def _dwb(n):
  return Dwb(DwbCfg(), n, "cpu")


def test_dwb_walks_to_a_goal_in_the_open_within_its_limits():
  poses, cmds, hit = _drive([], (4.0, 0.0), cycles=70, make=_dwb)
  assert not hit
  assert ((poses[-1, :, :2] - torch.tensor([4.0, 0.0])).norm(dim=-1) < 0.5).all()
  assert (cmds[..., 0] >= -1e-6).all() and (cmds[..., 0] <= 1.2 + 1e-6).all()
  assert (cmds[..., 1].abs() <= 0.5 + 1e-6).all()


def test_dwb_threads_an_opening_without_touching_it():
  wall = [(2.0, 2.3, 0.45, 3.0), (2.0, 2.3, -3.0, -0.45)]
  poses, _, hit = _drive(wall, (5.0, 0.0), cycles=80, make=_dwb)
  assert not hit
  assert ((poses[-1, :, :2] - torch.tensor([5.0, 0.0])).norm(dim=-1) < 0.5).all()


def test_dwb_turns_toward_a_goal_behind_it():
  poses, cmds, _ = _drive([], (-3.0, 0.0), cycles=40, make=_dwb)
  assert torch.isfinite(cmds).all()
  assert (poses[-1, :, 0] < -1.0).all()


def test_dwb_stops_when_every_twist_collides():
  # The robot's nose is already in a box: even standing still collides (in
  # the open, standing still is one of DWB's twists and always valid).
  box = [(0.1, 0.8, -0.6, 0.6)]
  cfg = DwbCfg()
  planner = Dwb(cfg, 1, "cpu")
  cmap = nav2_costmap(_local_heights(box, torch.zeros(1, 3)), RES, cfg.costmap)
  out = planner.step(cmap, RES, torch.tensor([[3.0, 0.0]]), torch.zeros(1, 3))
  assert bool(out["nav2_fail"][0]) and (out["cmd"] == 0).all()


def test_dwb_returns_each_twist_held_for_three_seconds_to_extra_critics():
  seen = {}

  def probe(data):
    seen["cv"] = data["cv"]
    return torch.zeros(data["cv"].shape[:2])

  planner = Dwb(DwbCfg(), 1, "cpu", extra_critics=[probe])
  cmap = torch.zeros(1, 160, 160)
  planner.step(cmap, RES, torch.tensor([[3.0, 0.0]]), torch.zeros(1, 3))
  cv = seen["cv"]
  assert cv.shape[2] == 30 and torch.equal(cv[:, :, 0], cv[:, :, -1])


def test_a_fast_rollout_cannot_step_through_a_thin_wall():
  # Nav2 checks every linear_granularity (5 cm); at 1.2 m/s a pose every
  # 0.1 s would hop 12 cm and could land either side of a wall's edge.
  cfg = DwbCfg()
  planner = Dwb(cfg, 1, "cpu")
  cmap = nav2_costmap(
    _local_heights([(1.0, 1.05, -3.0, 3.0)], torch.zeros(1, 3)), RES, cfg.costmap
  )
  out = planner.step(
    cmap, RES, torch.tensor([[3.0, 0.0]]), torch.tensor([[1.2, 0.0, 0.0]])
  )
  # Walking on at speed would cross the wall: it must slow or stop short.
  traj = out["path"][0, : round(1.7 / 0.1)]
  assert (traj[:, 0] + cfg.costmap.footprint[0] / 2 < 1.0).all()
