"""Tests for the fault impact helpers (frames, paths, grids, contact events)."""

import math
import os

import mujoco
import numpy as np
import pytest
import torch
from heat_bench.envs_mjlab.fault_impact import (
  FAULT_ELEVATION_SENSOR,
  GridSpec,
  SweepMap,
  assemble_risk,
  body_part,
  extract_contact_events,
  fire_step,
  geom_sample_points,
  integrate_command_path,
  path_deviation,
  rasterize_contacts,
  terrain_tile,
  tilt_angle,
  to_yaw_frame,
  window_risk,
)
from heat_bench.envs_mjlab.go2_eval_env_cfg import (
  add_joint_fault_injection,
  go2_eval_env_cfg,
  use_fault_terrain,
)
from heat_bench.scripts.render_fault_impact import pick_trials

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import RayCastSensor
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.lab_api.math import quat_from_euler_xyz


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def test_command_path_straight_and_turning():
  dt, k = 0.02, 200
  straight = integrate_command_path(
    torch.tensor([[[1.0, 0.5, 0.0]]]).expand(1, k, 3), dt
  )
  torch.testing.assert_close(straight[0, -1], torch.tensor([4.0, 2.0, 0.0]))
  # Constant turn: a circle of radius v / w.
  v, w = 1.0, 0.5
  turn = integrate_command_path(torch.tensor([[[v, 0.0, w]]]).expand(1, k, 3), dt)
  yaw = w * k * dt
  expected = torch.tensor([v / w * math.sin(yaw), v / w * (1 - math.cos(yaw)), yaw])
  torch.testing.assert_close(turn[0, -1], expected, atol=1e-4, rtol=0)


def test_path_deviation_wraps_heading():
  d = path_deviation(torch.tensor([1.0, 0.0, 3.1]), torch.tensor([0.0, 0.0, -3.1]))
  torch.testing.assert_close(d, torch.tensor([1.0, 0.0, 6.2 - 2 * math.pi]))


def test_yaw_frame_ignores_roll_and_pitch():
  # Yawed +90 deg and pitched/rolled: world +y is still forward, z still up.
  quat = quat_from_euler_xyz(
    torch.tensor([0.3]), torch.tensor([-0.4]), torch.tensor([math.pi / 2])
  )
  pts = torch.tensor([[[1.0, 3.0, -0.2]]])
  out = to_yaw_frame(pts, torch.tensor([[1.0, 2.0, 0.3]]), quat)
  torch.testing.assert_close(out[0, 0], torch.tensor([1.0, 0.0, -0.5]))


def test_grid_matches_ray_layout_and_rasterizes():
  grid = GridSpec(size=(1.0, 0.6), resolution=0.1)
  assert (grid.nx, grid.ny) == (11, 7)
  env_id = torch.tensor([0, 0, 1, 1])
  # Cell (iy=3, ix=8), twice; env 1: one in-grid at the corner, one outside.
  pos = torch.tensor(
    [[0.3, 0.0, 0.0], [0.31, 0.01, 0.0], [-0.5, -0.3, 0.0], [2.0, 0.0, 0.0]]
  )
  force = torch.tensor([10.0, 30.0, 5.0, 99.0])
  dt = torch.tensor([0.4, 0.2, 1.0, 0.1])
  risk, outside = rasterize_contacts(env_id, pos, force, dt, 2, grid)
  assert risk.shape == (2, 3, 7, 11)
  assert outside.tolist() == [0, 1]
  assert risk[0, :, 3, 8].tolist() == pytest.approx([2.0, 30.0, 0.2])
  assert risk[0, 0].sum() == 2 and risk[0, 2].isnan().sum() == 7 * 11 - 1
  assert risk[1, :, 0, 0].tolist() == pytest.approx([1.0, 5.0, 1.0])


def test_tilt_and_fire_step():
  g = torch.tensor([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
  torch.testing.assert_close(tilt_angle(g), torch.tensor([0.0, math.pi / 2]))
  # Same rule as inject_joint_fault: first k with k * dt >= t.
  t = torch.tensor([0.0, 0.02, 0.021, 3.0])
  k = fire_step(t, 0.02)
  assert (k.float() * 0.02 >= t).all() and ((k - 1).float() * 0.02 < t).all()


def test_terrain_tile_curriculum_layout():
  gen = ROUGH_TERRAINS_CFG  # 10 rows x 8 m; curriculum off -> 20 cols x 8 m.
  r, c = terrain_tile(
    torch.tensor([[-39.0, -79.0, 0.0], [1.0, 1.0, 0.0], [99.0, 0, 0]]), gen
  )
  assert r.tolist() == [0, 5, -1] and c.tolist() == [0, 10, -1]


def test_contact_events_timing_debounce_and_order():
  step_dt = 0.02
  active = np.zeros((1, 12, 2), dtype=bool)
  force = np.zeros((1, 12, 2), dtype=np.float32)
  # Part 0: steps 2-3, a 2-step gap (merged by debounce), then 6; then 10.
  for s in (2, 3, 6, 10):
    active[0, s, 0] = True
    force[0, s, 0] = 10.0 * s
  # Part 1: starts on step 2 too, harder hit -> ranked first.
  active[0, 2, 1] = True
  force[0, 2, 1] = 500.0
  dt = (np.arange(12) - 1) * step_dt  # Step 1 is the fault.
  ev = extract_contact_events(active, force, dt[None], step_dt, debounce_steps=2)
  assert ev["part"].tolist() == [1, 0, 0]
  assert ev["order"].tolist() == [0, 1, 2]
  assert ev["onset_step"].tolist() == [2, 2, 10]
  assert ev["end_step"].tolist() == [2, 6, 10]
  np.testing.assert_allclose(ev["onset_dt"], [0.02, 0.02, 0.18], atol=1e-6)
  np.testing.assert_allclose(ev["duration_s"], [0.02, 0.1, 0.02], atol=1e-6)
  np.testing.assert_allclose(ev["peak_force_n"], [500.0, 60.0, 100.0])
  np.testing.assert_allclose(ev["impulse_ns"][1], (20 + 30 + 60) * step_dt)


def test_body_part_merges_capsules():
  assert body_part("FR_thigh_collision2") == "FR_thigh"
  assert body_part("trunk_collision") == "trunk"


def test_config_command_ranges_and_terrain():
  cfg = go2_eval_env_cfg(play=True)
  add_joint_fault_injection(
    cfg,
    ("FR_calf_joint",),
    command_vel=None,
    command_ranges=((0.3, 1.0), (-0.2, 0.2), (-0.5, 0.5)),
    terrain="stairs",
    elevation_map=True,
  )
  twist = cfg.commands["twist"]
  assert twist.ranges.lin_vel_y == (-0.2, 0.2)  # type: ignore[attr-defined]
  assert twist.resampling_time_range[0] >= 1e9
  assert (
    cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None
  )
  assert cfg.scene.terrain.terrain_generator.curriculum
  assert cfg.scene.terrain.max_init_terrain_level is None
  assert any(s.name == FAULT_ELEVATION_SENSOR for s in cfg.scene.sensors or ())
  with pytest.raises(ValueError, match="not both"):
    add_joint_fault_injection(
      go2_eval_env_cfg(play=True),
      ("FR_calf_joint",),
      command_ranges=((0.0, 1.0), (0.0, 0.0), (0.0, 0.0)),
    )


@pytest.mark.slow
def test_elevation_sensor_in_real_env():
  cfg = go2_eval_env_cfg(play=True)
  add_joint_fault_injection(
    cfg, ("FR_calf_joint",), modes=("none",), terrain="rough", elevation_map=True
  )
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  try:
    env.reset()
    env.step(torch.zeros(env.action_space.shape, device=env.device))
    sensor = env.scene[FAULT_ELEVATION_SENSOR]
    assert isinstance(sensor, RayCastSensor)
    grid = GridSpec(size=(4.0, 4.0), resolution=0.1)
    assert sensor.data.hit_pos_w.shape == (2, grid.ny * grid.nx, 3)
    # Standing robot: terrain is ~a trunk height below the root link.
    root_z = env.scene["robot"].data.root_link_pos_w[:, 2:3]
    centre = sensor.data.hit_pos_w[:, (grid.ny // 2) * grid.nx + grid.nx // 2, 2:3]
    assert ((root_z - centre) > 0.1).all()
    # Rays start above the trunk, so every cell sees terrain.
    assert (sensor.data.distances > 0).all()
  finally:
    env.close()


def test_sweep_marks_footprint_first_time_and_clearance():
  grid = GridSpec(size=(1.0, 1.0), resolution=0.1)
  sweep = SweepMap(1, grid, "cpu")
  terrain = torch.zeros(1, grid.ny, grid.nx)
  terrain[0, 5, 5] = 0.05  # A bump under the origin.
  radius = torch.tensor([0.05])
  rows = torch.tensor([0])
  # A 5 cm sphere 12 cm up at the origin, then the same sphere 0.3 m ahead
  # resting on the ground.
  sweep.update(
    rows, torch.tensor([[[0.0, 0.0, 0.12]]]), radius, terrain, torch.tensor([0.1])
  )
  sweep.update(
    rows, torch.tensor([[[0.3, 0.0, 0.05]]]), radius, terrain, torch.tensor([0.4])
  )
  occupied, first, clear = sweep.result()[0]
  # Reach is radius + half a cell = 0.1 m: a plus shape of 5 cells each.
  assert occupied.sum() == 10
  assert first[5, 5] == pytest.approx(0.1) and first[5, 8] == pytest.approx(0.4)
  assert clear[5, 5] == pytest.approx(0.12 - 0.05 - 0.05)
  assert clear[5, 8] == pytest.approx(0.0, abs=1e-6)
  assert first[0, 0].isnan() and clear[0, 0].isnan()


def test_geom_sample_points_cover_capsule_and_box():
  types = np.array([mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_BOX])
  sizes = np.array([[0.02, 0.1, 0.0], [0.13, 0.09, 0.05]])
  idx, off, rad = geom_sample_points(types, sizes, np.array([0.12, 0.17]))
  assert idx.tolist() == [0] * 5 + [1] * 9
  torch.testing.assert_close(off[[0, 4], 2], torch.tensor([-0.1, 0.1]))
  assert rad[0] == pytest.approx(0.02) and rad[5] == pytest.approx(0.05)
  assert off[5:, 0].abs().max() == pytest.approx(0.13)


def test_assemble_risk_channels():
  contacts = torch.tensor(
    [[[[0.0, 3.0]], [[float("nan"), 50.0]], [[float("nan"), 0.2]]]]
  )
  risk = assemble_risk(contacts, torch.tensor([[[0.0, 1.0]]]))
  assert risk.shape == (1, 5, 1, 2)
  assert risk[0, :, 0, 1].tolist() == pytest.approx([1.0, 50.0, 0.2, 1.0, 3.0])
  assert risk[0, 0, 0, 0] == 0.0


def test_stairs_up_preset_is_one_climbing_column():
  cfg = go2_eval_env_cfg(play=True)
  use_fault_terrain(cfg, "stairs_up")
  assert cfg.scene.terrain is not None
  gen = cfg.scene.terrain.terrain_generator
  assert gen is not None and gen.curriculum and list(gen.sub_terrains) == ["stairs_up"]
  with pytest.raises(ValueError, match="Unknown terrain"):
    use_fault_terrain(cfg, "ramp")


def test_window_risk_splits_by_time_since_fault():
  grid = GridSpec(size=(1.0, 1.0), resolution=0.1)
  pos = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3, 0.0, 0.0]])
  # Contacts at 0.5 s (origin) and 1.5 s, 1.7 s (0.3 m ahead).
  contacts = (
    torch.tensor([0, 0, 0]),
    pos,
    torch.tensor([10.0, 40.0, 20.0]),
    torch.tensor([0.5, 1.5, 1.7]),
  )
  # One impact started at 1.5 s, ahead.
  events = (torch.tensor([0]), pos[1:2], torch.tensor([40.0]), torch.tensor([1.5]))
  risk = window_risk(contacts, events, torch.tensor([0.0, 1.0, 2.0]), 1, grid)
  assert risk.shape == (1, 2, 5, grid.ny, grid.nx)
  touched, peak, first, impacts, steps = risk[0, :, :, 5, 5].T
  assert touched.tolist() == [1.0, 0.0]
  w1 = risk[0, 1, :, 5, 8]
  assert w1.tolist() == pytest.approx([1.0, 40.0, 1.5, 1.0, 2.0])
  assert risk[0, 0, 3].sum() == 0  # No impact started in [0, 1).


def test_pick_trials_prefers_falls_with_most_impact_cells_per_mode():
  risk = np.zeros((5, 5, 2, 2))
  risk[1, 0, 0, :] = 1.0  # Trial 1: 2 touched cells.
  risk[2, 0, 0, 0] = 1.0  # Trial 2: 1 touched cell.
  out = {
    "env_kept": np.array([True, True, True, True, False]),
    "fault_mode": np.array([1, 1, 1, 2, 1]),
    "fell": np.array([False, True, True, False, True]),
    "risk_fy": risk,
    "risk_channels": np.array(
      ["touched", "peak_force_n", "first_contact_dt_s", "impacts", "contact_steps"]
    ),
  }
  # Mode 1: the fall with most cells, then a survivor; mode 2: its survivor.
  assert pick_trials(out, per_mode=2) == [1, 0, 3]
