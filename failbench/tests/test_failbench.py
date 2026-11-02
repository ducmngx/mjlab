"""Tests for the FailBench fault sampler, LiDAR visibility and dataset."""

import json
import math
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import mujoco
import numpy as np
import pytest
import torch
from failbench.commands import (
  PREFAULT_MODES,
  GoalVelocityCommand,
  GoalVelocityCommandCfg,
  PrefaultCfg,
  prefault_still,
)
from failbench.data.build import (
  SPLITS,
  BuildConfig,
  FallMonitor,
  _split,
  fall_step,
  no_return_step,
)
from failbench.data.dataset import FailBenchDataset
from failbench.faults import (
  FAULT_TERM,
  JOINT_NAMES,
  LOCK,
  NOMINAL,
  TORQUE,
  FaultFamily,
  FaultMixture,
  encode_fault,
  faulted_joint,
  inject_kinodynamic_fault,
  sample_fault_params,
)
from failbench.goals import route_progress, sample_goal
from failbench.model.losses import (
  LossWeights,
  hazard_nll,
  impact_loss,
  impact_metrics,
)
from failbench.model.model import ImpactModel, ImpactModelCfg, rotate_xy
from failbench.plans import PlanCfg, SegmentPlan
from failbench.scene_course import (
  CLUSTER_COURSE,
  CLUSTER_COURSE_TERRAIN,
  CLUSTER_SWAP,
  CLUTTER_FIELD,
  CLUTTER_SWAP,
  GAP_COURSE,
  GAP_SWAP,
  PASS_COURSE,
  PASS_SWAP,
)
from failbench.scripts.evaluate import EvalConfig, evaluate_split
from failbench.scripts.run_mppi import _touched_severity
from failbench.sensing import LidarCfg, sensor_poses, visible_cells
from failbench.severity import DEFAULT_SEVERITY, SEVERITY, geom_severity


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def test_sampled_families_fault_one_joint_each():
  torch.manual_seed(0)
  params, family = sample_fault_params(4000, 12, FaultMixture(), "cpu")
  touched = ~torch.isclose(params, torch.tensor(NOMINAL)).all(-1)  # [N, J]
  per_env = touched.sum(-1)
  assert (per_env[family == FaultFamily.NOMINAL] == 0).all()
  assert (per_env[family != FaultFamily.NOMINAL] == 1).all()
  joint = faulted_joint(params)
  assert ((joint >= 0) == (family != FaultFamily.NOMINAL)).all()
  rows = torch.arange(len(family))
  hit = params[rows, joint.clamp(min=0)]
  lock, weak = family == FaultFamily.LOCK, family == FaultFamily.WEAK
  assert (hit[lock, LOCK] == 1).all() and (hit[lock, TORQUE] == 1).all()
  assert (hit[weak, LOCK] == 0).all() and (hit[weak, TORQUE] <= 0.5).all()
  limp = (hit[weak, TORQUE] == 0).float().mean()
  assert abs(limp - FaultMixture().limp_prob) < 0.05
  freq = torch.bincount(family, minlength=3).float() / len(family)
  torch.testing.assert_close(
    freq, torch.tensor(FaultMixture().probs), atol=0.03, rtol=0
  )


def test_healthy_joint_encodes_to_reference():
  enc = encode_fault(torch.tensor(NOMINAL).repeat(2, 12, 1))
  assert enc.shape == (2, 24)
  torch.testing.assert_close(enc[0, :2], torch.tensor([1.0, 0.0]))


def _run_plan(hold: bool, steps: int = 350, n: int = 512) -> tuple:
  torch.manual_seed(0)
  plan = SegmentPlan(n, PlanCfg(), "cpu")
  ids = torch.arange(n)
  c0 = torch.tensor([0.8, 0.0, 0.1]).repeat(n, 1)
  goal = torch.tensor([0.0, 5.0]).repeat(n, 1)  # 90 deg to the left
  plan.start(ids, c0, torch.zeros(n, 3), goal, hold=torch.full((n,), hold))
  cmds = []
  for _ in range(steps):
    plan.step(ids, 0.02)
    cmds.append(plan.cmd.clone())
  return plan, torch.stack(cmds, dim=1)


def test_hold_plan_keeps_the_fault_time_command():
  _, cmds = _run_plan(hold=True)
  assert (cmds == torch.tensor([0.8, 0.0, 0.1])).all()


def test_segment_plans_stay_in_limits_and_pull_toward_the_goal():
  plan, cmds = _run_plan(hold=False)
  lim = torch.tensor(PlanCfg().limits)
  assert (cmds[..., 0] >= 0).all() and (cmds.abs() <= lim + 1e-6).all()
  # vx ramps at most max_accel per step.
  assert (cmds[:, 1:, 0] - cmds[:, :-1, 0]).abs().max() <= 0.02 + 1e-6
  assert (plan.seg_count >= 2).all()
  # The goal is to the left: plans mostly end up turned toward it.
  assert plan.pose[:, 2].median() > math.pi / 4


def test_lane_plans_return_to_the_centre_line():
  torch.manual_seed(0)
  n = 512
  plan = SegmentPlan(n, PlanCfg(), "cpu")
  ids = torch.arange(n)
  # Start 0.4 m left of a lane along +x through (10, 0), heading +x.
  pose = torch.tensor([0.0, 0.4, 0.0]).repeat(n, 1)
  plan.start(
    ids,
    torch.tensor([0.8, 0.0, 0.0]).repeat(n, 1),
    pose,
    torch.tensor([10.0, 0.0]).repeat(n, 1),
    hold=torch.zeros(n, dtype=torch.bool),
    lane=torch.ones(n, dtype=torch.bool),
    bearing=torch.zeros(n),
  )
  for _ in range(250):
    plan.step(ids, 0.02)
  assert plan.pose[:, 1].abs().median() < 0.15


def _fake_batch(b: int = 3, t: int = 50) -> dict[str, torch.Tensor]:
  cells = 64
  upright = torch.ones(b, t, dtype=torch.bool)
  upright[0, 20:] = False  # Sample 0 falls in step 20.
  return {
    "map": torch.stack(
      [torch.full((b, cells, cells), -0.3), torch.ones(b, cells, cells)], dim=1
    ),
    "state": torch.randn(b, 33),
    "him_latent": torch.randn(b, 16),
    "history": torch.randn(b, 20, 52),
    "latent_future": torch.randn(b, t, 16),
    "fault": torch.tensor(NOMINAL).repeat(b, 12),
    "cmd_seq": torch.rand(b, t, 3),
    "cmd_path": torch.cumsum(torch.rand(b, t, 3) * 0.1, dim=1),
    "pose": torch.cumsum(torch.rand(b, t, 3) * 0.1, dim=1),
    "dv": torch.randn(b, t, 3) * 0.1,
    "upright": upright,
    "valid": torch.ones(b, t, dtype=torch.bool),
    "collision": torch.zeros(b, t, dtype=torch.bool),
    "contact_offmap": torch.zeros(b, t, dtype=torch.bool),
    "contact": torch.rand(b, t, cells, cells) > 0.999,
  }


def _unit_stats() -> dict:
  dims = {
    "state": 33,
    "him_latent": 16,
    "history": 52,
    "cmd_seq": 3,
    "cmd_path": 3,
    "map": 1,
  }
  return {k: {"mean": [0.0] * n, "std": [1.0] * n} for k, n in dims.items()}


def test_impact_model_shapes_and_plan_batching():
  torch.manual_seed(0)
  cfg = ImpactModelCfg(
    d_model=32, context_layers=1, decoder_layers=1, contact_heads=True
  )
  model = ImpactModel(cfg, _unit_stats()).eval()
  batch = _fake_batch()
  pred = model(batch)
  assert pred["drift"].shape == (3, 50, 3)
  assert pred["contact_logit"].shape == (3, 5, 64, 64)
  # P(fallen by t) never decreases.
  assert (pred["p_fallen"].diff(dim=1) >= -1e-6).all()
  total, terms = impact_loss(pred, batch, LossWeights(), cfg.steps_per_map)
  assert torch.isfinite(total) and set(terms) >= {"drift", "fall", "contact"}
  # One context scores many plans.
  ctx = model.encode({k: v[:1] for k, v in batch.items()})
  many = model.decode(ctx, torch.rand(8, 50, 3), torch.rand(8, 50, 3), False)
  assert many["drift"].shape == (8, 50, 3) and "contact_logit" not in many
  # Changing the plan after step 30 leaves steps up to 30 unchanged.
  seq, path = batch["cmd_seq"].clone(), batch["cmd_path"].clone()
  seq[:, 31:], path[:, 31:] = 0.5, 0.5
  late = model.decode(model.encode(batch), seq, path, False)
  assert torch.allclose(late["drift"][:, :31], pred["drift"][:, :31], atol=1e-5)
  assert not torch.allclose(late["drift"][:, 31:], pred["drift"][:, 31:])


def test_hazard_nll_prefers_the_true_fall_step():
  upright = torch.ones(1, 50, dtype=torch.bool)
  upright[0, 20:] = False
  valid = torch.ones(1, 50, dtype=torch.bool)
  right = torch.full((1, 50), -6.0)
  right[0, 20] = 6.0
  wrong = torch.full((1, 50), -6.0)
  wrong[0, 40] = 6.0
  assert hazard_nll(right, upright, valid) < hazard_nll(wrong, upright, valid)


def _visible(height: torch.Tensor, lidar: LidarCfg) -> torch.Tensor:
  pos, axis = sensor_poses(
    torch.zeros(1, 1, 3), torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]), lidar
  )
  return visible_cells(
    height[None], pos, axis, torch.zeros(1, 1, 3), torch.zeros(1, 1), 0.05, lidar
  )[0]


def test_lidar_sees_flat_ground_ahead_but_not_under_the_trunk():
  lidar = LidarCfg()
  seen = _visible(torch.full((64, 64), -0.3), lidar)
  centre = (64 - 1) / 2
  ahead = seen[int(centre), int(centre) + 20]  # 1 m ahead
  under = seen[int(centre), int(centre) - 2]  # under the trunk
  assert ahead and not under


def test_lidar_cannot_see_the_tread_below_a_step_down():
  # Ground at -0.3 m up to 0.6 m ahead, then a 0.3 m drop.
  height = torch.full((64, 64), -0.3)
  x = (torch.arange(64) - 31.5) * 0.05
  height[:, x > 0.6] = -0.6
  seen = _visible(height, LidarCfg())
  row = 32
  just_below = seen[row, (x > 0.62) & (x < 0.8)]
  far_below = seen[row, x > 1.4]
  assert not just_below.any()
  assert far_below.all()


def test_dataset_rasterises_csr_contacts(tmp_path):
  shard = tmp_path / "s0"
  shard.mkdir()
  n, nb, cells = 2, 50, 64
  per_sample = {
    "state": np.zeros((n, 33), np.float32),
    "state_true_vel": np.zeros((n, 3), np.float32),
    "him_latent": np.zeros((n, 16), np.float32),
    "history": np.zeros((n, 20, 36), np.float32),
    "cmd_seq": np.zeros((n, nb, 3), np.float32),
    "fault": np.zeros((n, 24), np.float32),
    "dv": np.zeros((n, nb, 3), np.float32),
    "pose": np.zeros((n, nb, 3), np.float32),
    "cmd_path": np.zeros((n, nb, 3), np.float32),
    "upright": np.ones((n, nb), bool),
    "collision": np.zeros((n, nb), bool),
    "valid": np.ones((n, nb), bool),
    "contact_offmap": np.zeros((n, nb), bool),
    "fault_family": np.zeros(n, np.uint8),
    "plan_hold": np.zeros(n, bool),
    "offset_s": np.zeros(n, np.float32),
    "split": np.zeros(n, np.uint8),
    "fallen_at_start": np.zeros(n, bool),
    "map_clean": np.full((n, cells, cells), -0.3, np.float16),
    "known_fused": np.ones((n, cells, cells), bool),
    "contact_ptr": np.array([0, 0, 2]),
    "contact_bin": np.array([3, 7], np.uint8),
    "contact_ix": np.array([10, 20], np.uint8),
    "contact_iy": np.array([30, 40], np.uint8),
    "contact_part": np.array([0, 0], np.uint8),
  }
  for k, v in per_sample.items():
    np.save(shard / f"{k}.npy", v)
  meta = {"num_bins": nb, "map_cells": cells, "terrain_id": 0}
  (shard / "meta.json").write_text(json.dumps(meta))
  ds = FailBenchDataset(tmp_path, SPLITS[0], seed=0)
  assert len(ds) == 2
  assert not ds[0]["contact"].any()
  c = ds[1]["contact"]
  assert c.sum() == 2 and c[3, 30, 10] and c[7, 40, 20]
  assert ds[1]["map"].shape == (2, cells, cells)


@pytest.mark.slow
def test_fault_writes_and_plan_starts_in_real_env():
  from failbench.env_cfg import failbench_env_cfg

  from mjlab.envs import ManagerBasedRlEnv

  device = get_test_device()
  cfg = failbench_env_cfg("Go2-HIM-Rough", "rough", 0, fault_time_s=(0.2, 0.2))
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=device)
  env.reset()
  fault = env.event_manager.get_term_cfg(FAULT_TERM).func
  command = env.command_manager.get_term("twist")
  assert isinstance(command, GoalVelocityCommand)
  # Env 0: joint 2 (FL calf; asset order is FL, FR, RL, RR) locked; env 1:
  # every joint limp.
  fault.params[:] = torch.tensor(NOMINAL, device=device)
  fault.params[0, 2, LOCK] = 1.0
  fault.params[1, :, TORQUE] = 0.0
  robot = env.scene["robot"]
  act = torch.zeros(2, 12, device=device)
  q_at_fault = None
  for _ in range(80):
    env.step(act)
    if q_at_fault is None and fault.faulted.all():
      # On the fault step the plan starts at the follower's command.
      q_at_fault = fault.fault_qpos[0, 2].item()
      assert command.started.all()
      torch.testing.assert_close(command.plan.cmd, command.c0)
  assert q_at_fault is not None
  assert abs(robot.data.joint_pos[0, 2].item() - q_at_fault) < 0.02
  assert robot.data.actuator_force[1].abs().max().item() < 1e-4
  # Open loop: the predicted pose integrates the plan, not the robot.
  assert (command.plan.pose[:, :2] != robot.data.root_link_pos_w[:, :2]).any()
  assert math.isfinite(robot.data.joint_pos.sum().item())
  env.close()


@pytest.mark.slow
def test_the_driver_commits_on_the_fault_step_in_real_env():
  from failbench.driver import PlannerDriveCfg
  from failbench.env_cfg import failbench_env_cfg
  from failbench.mppi.inputs import deadband

  from mjlab.envs import ManagerBasedRlEnv

  device = get_test_device()
  # 1.23 s: after the 1 s warm-up and between planning cycles (every 0.1 s).
  cfg = failbench_env_cfg(
    "Go2-HIM-Rough", "rough", 0, fault_time_s=(1.23, 1.23), driver=PlannerDriveCfg()
  )
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=device)
  env.reset()
  fault = env.event_manager.get_term_cfg(FAULT_TERM).func
  command = env.command_manager.get_term("twist")
  assert isinstance(command, GoalVelocityCommand) and command.driver is not None
  # Phantoms lie along each trial's own spawn-to-goal line (the reset reads
  # the new spawn, not the stale body pose).
  robot = env.scene["robot"]
  start = robot.data.data.qpos[:, robot.indexing.free_joint_q_adr[:2]]
  goal = command.goal_w[:, :2]
  u = (goal - start) / (goal - start).norm(dim=-1, keepdim=True)
  boxes = command.driver.boxes
  real = boxes[..., 2] > 0
  rel = boxes[..., :2] - start[:, None]
  along = (rel * u[:, None]).sum(-1)
  side = (rel[..., 0] * -u[:, None, 1] + rel[..., 1] * u[:, None, 0]).abs()
  assert (along[real] > 0.7).all() and (along[real] < 4.6).all()
  assert (side[real] < 1.51).all()
  act = torch.zeros(2, 12, device=device)
  seen = False
  for _ in range(100):
    env.step(act)
    if not seen and fault.faulted.all():
      seen = True
      # On the fault step itself the first committed command runs.
      assert command.started.all()
      torch.testing.assert_close(
        command.command, deadband(command.committed[:, 0], 0.2)
      )
  assert seen
  env.close()


def _after(seed: int, draws: int) -> torch.Generator:
  """A generator seeded with ``seed`` that has drawn ``draws`` uniforms."""
  g = torch.Generator().manual_seed(seed)
  torch.rand(draws, generator=g)
  return g


def test_prefault_modes_stand_set_off_and_stop_around_the_fault():
  m = {k: i for i, k in enumerate(PREFAULT_MODES)}
  mode = torch.tensor([m["walk"], m["stand"], m["start"], m["slow"], m["stopped"]])
  lead = torch.full((5,), 1.0)
  fault_time = torch.full((5,), 4.0)

  def still(t: float) -> list[bool]:
    return prefault_still(torch.full((5,), t), fault_time, mode, lead, 1.0).tolist()

  assert still(0.5) == [True] * 5  # warm-up
  assert still(2.0) == [False, True, True, False, False]
  assert still(3.5) == [False, True, False, False, True]

  # The draws follow the mix, and only slow trials cruise slowly.
  cmd = GoalVelocityCommand.__new__(GoalVelocityCommand)
  pc = PrefaultCfg(mix=(1, 1, 1, 1, 1), slow_speed=(0.2, 0.3))
  cmd.goal_cfg = GoalVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    ranges=GoalVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    ),
    prefault=pc,
  )
  env = type("Env", (), {"device": "cpu"})()
  cmd._env = env  # pyright: ignore[reportAttributeAccessIssue]
  n = 2000
  cmd.speed, cmd.lead = torch.zeros(n), torch.zeros(n)
  cmd.mode = torch.zeros(n, dtype=torch.long)
  cmd.started = torch.ones(n, dtype=torch.bool)
  cmd.use_external = torch.ones(n, dtype=torch.bool)
  cmd.vel_command_b = torch.ones(n, 3)
  cmd.driver = None
  torch.manual_seed(0)
  cmd._resample_command(torch.arange(n))
  share = torch.bincount(cmd.mode, minlength=5) / n
  assert (share - 0.2).abs().max() < 0.04
  slow = cmd.mode == m["slow"]
  assert cmd.speed[slow].max() <= 0.3 and cmd.speed[~slow].min() >= 0.5
  assert not cmd.started.any() and (cmd.vel_command_b == 0).all()

  # All walk (the default) draws only the cruise speeds, as before the modes,
  # so seeded runs keep their faults.
  cmd.goal_cfg = replace(cmd.goal_cfg, prefault=PrefaultCfg())
  torch.manual_seed(1)
  cmd._resample_command(torch.arange(n))
  lo, hi = cmd.goal_cfg.speed
  torch.manual_seed(1)
  torch.testing.assert_close(cmd.speed, lo + (hi - lo) * torch.rand(n))
  assert (cmd.mode == 0).all()
  torch.testing.assert_close(torch.rand(3), torch.rand(3, generator=_after(1, n)))


def test_a_waypoint_passed_outside_its_radius_still_counts():
  term = sample_goal.__new__(sample_goal)
  term.start = torch.zeros(1, 3)
  term.waypoints = torch.tensor([[[2.0, 0.0, 0.0], [2.0, 3.0, 0.0]]])
  term.index = torch.zeros(1, dtype=torch.long)
  term.goal_w = term.waypoints[:, 0]
  term.advance(torch.tensor([[1.5, 0.8]]), radius=0.4)  # Beside, not past.
  assert term.index[0] == 0
  term.advance(torch.tensor([[2.1, 0.8]]), radius=0.4)  # Past, 0.8 m off.
  assert term.index[0] == 1 and bool(term.final[0])
  term.advance(torch.tensor([[5.0, 5.0]]), radius=0.4)  # The last one stays.
  assert term.index[0] == 1


def test_a_touch_takes_the_severity_of_what_is_under_it_far_from_the_origin():
  # World coordinates of a far tile; on GPU in TF32 a matmul-based distance
  # matched floor contacts to glass a metre away.
  dev = "cuda" if torch.cuda.is_available() else "cpu"
  base = torch.tensor([49.3, -7.4], device=dev)
  hits = base + torch.tensor([[0.0, 0.0], [1.2, -0.8], [0.03, 0.0]], device=dev)
  rays = SimpleNamespace(
    hit_pos_w=torch.cat([hits, torch.zeros(3, 1, device=dev)], -1)[None],
    hit_geom_id=torch.tensor([[0, 1, 2]], device=dev),
  )
  geom_sev = torch.tensor([0.0, 25.0, 5.0], device=dev)  # Floor, glass, crate.
  contact = torch.zeros(1, 2, 3, device=dev)
  contact[0, :, :2] = base
  touch = torch.tensor([[True, False]], device=dev)
  assert _touched_severity(touch, contact, rays, geom_sev).item() == 5.0  # type: ignore[arg-type]


def test_geom_names_map_to_severity():
  spec = mujoco.MjSpec()
  names = ("course_glass_3", "course_wall_7", "course_floor_0", "", "terrain_12")
  for name in names:
    spec.worldbody.add_geom(name=name, size=(0.1, 0.1, 0.1))
  sev = geom_severity(spec.compile()).tolist()
  # The ground itself (rough tiles, stairs) is no damage to touch.
  expect = [SEVERITY["glass"], SEVERITY["wall"], 0.0, DEFAULT_SEVERITY, 0.0, 0.0]
  assert sev == expect


@pytest.mark.parametrize(
  ("scene", "twin"),
  [
    (CLUSTER_COURSE, CLUSTER_SWAP),
    (GAP_COURSE, GAP_SWAP),
    (PASS_COURSE, PASS_SWAP),
    (CLUTTER_FIELD, CLUTTER_SWAP),
  ],
)
def test_scenes_keep_the_line_clear_and_their_twins_swap_glass_for_crates(scene, twin):
  def build(cfg, difficulty):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = replace(cfg, size=CLUSTER_COURSE_TERRAIN.size)
    out = cfg.function(difficulty, spec, np.random.default_rng(0))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = [model.geom(i).name.split("_")[1] for i in range(model.ngeom)]
    pos = data.geom_xpos[:, :2] - out.origin[:2]
    return names, pos, model.geom_size[:, :2]

  for difficulty in (0.0, 1.0):
    names, pos, size = build(scene, difficulty)
    # A robot-wide strip (0.31 m) from the spawn to the goal meets nothing
    # but the pass-around pillar.
    hit = {
      n
      for n, (x, y), (sx, sy) in zip(names, pos, size, strict=True)
      if n != "floor" and x + sx > 0 and x - sx < scene.goal_x and abs(y) - sy < 0.155
    }
    assert hit == ({"wall"} if scene is PASS_COURSE else set())
    # Every glass piece has crates where it stood in the twin, and back.
    tnames, tpos, _ = build(twin, difficulty)
    for a, pa, b, pb in ((names, pos, tnames, tpos), (tnames, tpos, names, pos)):
      glass = pa[[n == "glass" for n in a]]
      boxes = pb[[n == "box" for n in b]]
      assert len(glass) and len(boxes)
      gap = np.linalg.norm(glass[:, None] - boxes[None], axis=-1).min(1)
      assert (gap < 0.3).all()


def test_clutter_fields_differ_per_tile_and_keep_their_corridor():
  def objects(seed, difficulty):
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = replace(CLUTTER_FIELD, size=CLUSTER_COURSE_TERRAIN.size)
    out = cfg.function(difficulty, spec, np.random.default_rng(seed))
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    keep = [
      model.geom(i).name.split("_")[1] in ("box", "glass") for i in range(model.ngeom)
    ]
    pos = data.geom_xpos[keep, :2] - out.origin[:2]
    return pos, model.geom_size[keep, :2]

  layouts = [objects(seed, 1.0) for seed in range(4)]
  assert len({len(p) for p, _ in layouts}) > 1 or not np.allclose(
    layouts[0][0], layouts[1][0]
  )
  for pos, size in layouts:
    # 0.8 m corridor at difficulty 1: every near face 0.4 m or more off the line.
    assert (np.abs(pos[:, 1]) - size[:, 1] >= 0.4 - 1e-6).all()
  wide, size = objects(0, 0.0)
  assert (np.abs(wide[:, 1]) - size[:, 1] >= 0.8 - 1e-6).all()


def test_v2_model_drops_the_contact_heads_and_learns_the_no_return_label():
  torch.manual_seed(0)
  cfg = ImpactModelCfg(d_model=32, context_layers=1, decoder_layers=1)
  model = ImpactModel(cfg, _unit_stats())
  batch = _fake_batch()
  batch["safe"] = batch["upright"].clone()
  batch["safe"][0, 12:] = False  # Past the point of no return in step 12.
  pred = model(batch)
  assert {"collision_logit", "offmap_logit", "contact_logit"}.isdisjoint(pred)
  assert pred["latent"].shape == (3, 50, 16)
  total, terms = impact_loss(pred, batch, LossWeights(fall_pos=5.0), 10, "safe")
  assert set(terms) == {"drift", "dv", "fall", "latent"}
  # Weight 0 drops the latent target (the ablation).
  _, terms0 = impact_loss(pred, batch, LossWeights(latent=0.0), 10, "safe")
  assert "latent" not in terms0
  total.backward()  # One training step's gradient flows.
  m = impact_metrics(pred, batch, fall_label="safe")
  assert m["fell"].tolist() == [True, False, False]
  assert m["fell_soon"].tolist() == [False, False, False]  # Step 12 > 1 s.
  # A phase 1 config (saved before contact_heads) still has them, no latent
  # head, and reads the first 36 of a newer shard's history.
  old = ImpactModelCfg.from_saved({"d_model": 32, "history_dim": 36})
  assert old.contact_heads and not old.latent_head
  stats = _unit_stats() | {"history": {"mean": [0.0] * 36, "std": [1.0] * 36}}
  assert ImpactModel(old, stats)(batch)["drift"].shape == (3, 50, 3)
  # Trained anew on 52-wide shards' stats, it reads their first 36.
  assert ImpactModel(old, _unit_stats())(batch)["drift"].shape == (3, 50, 3)


def test_no_return_is_when_the_robot_leaves_its_walk_for_good():
  t = 200
  tilt = torch.full((3, t), 0.1)
  height = torch.full((3, t), 0.30)
  # 0: wobbles at 120 (recovers), tips from 150, on the ground at 170.
  tilt[0, 120:125] = 0.6
  tilt[0, 150:] = torch.linspace(0.5, 1.4, t - 150)
  # 1: sinks from 130 and stays low, never "fallen" by tilt, down at 160.
  height[1, 130:] = 0.2
  height[1, 160:] = 0.1
  # 2: never falls.
  fall = torch.tensor([170, 160, t])
  nr = no_return_step(tilt, height, fall, 100, 15.0, 0.06, 50, 75)
  assert nr.tolist() == [150, 130, t]
  # Capped: limping on tilted for long isn't falling yet (1 down at 160 is
  # no earlier than 160 - 20).
  nr = no_return_step(tilt, height, fall, 100, 15.0, 0.06, 50, 20)
  assert nr.tolist() == [150, 140, t]


def test_drift_is_spread_in_the_path_frame_with_heavy_tails():
  torch.manual_seed(0)
  cfg = ImpactModelCfg(d_model=32, context_layers=1, decoder_layers=1)
  stats = _unit_stats() | {"history": {"mean": [0.0] * 52, "std": [1e-4] * 52}}
  model = ImpactModel(cfg, stats)
  # The floor no longer squashes HIM's tiny latent spread.
  one_spread = model.norm("history", torch.full((52,), 1e-4))
  assert torch.allclose(one_spread, torch.ones(52))
  batch = _fake_batch()
  batch["cmd_path"][..., 2] = 0.7  # Heading 0.7 rad off the start frame.
  pred = model(batch)
  assert torch.allclose(
    pred["drift"], rotate_xy(pred["drift_local"], pred["drift_yaw"])
  )
  assert torch.allclose(pred["drift_yaw"], batch["cmd_path"][..., 2])
  # A throw 20 spreads off costs far less under the Student-t than a
  # Gaussian: one rare big throw doesn't drag the spread of the rest.
  far = dict(batch, pose=batch["cmd_path"] + 20 * pred["drift_std"].detach())
  far["pose"] = far["pose"] + pred["drift"].detach()
  w = LossWeights(latent=0.0)
  gauss = impact_loss(pred, far, w, 10, drift_nu=0.0)[1]["drift"]
  student = impact_loss(pred, far, w, 10, drift_nu=4.0)[1]["drift"]
  assert student < gauss / 10
  # Phase 1: start frame, Gaussian, the old floor.
  old = ImpactModelCfg.from_saved({})
  assert (old.drift_frame, old.drift_nu, old.norm_min_std) == ("start", 0.0, 1e-3)


def test_a_fall_is_down_for_good_with_the_trunk_on_the_ground():
  t = 200
  tilt = torch.full((4, t), 0.1)
  height = torch.full((4, t), 0.30)
  contact = torch.zeros(4, t, dtype=torch.bool)
  # 0: stumbles over 60 deg with a knock at 120, back upright by 130.
  tilt[0, 120:126] = 1.2
  contact[0, 121] = True
  # 1: trunk read 0.12 m over a stair edge at 120, no contact, walks on.
  height[1, 120:124] = 0.12
  # 2: tips over at 140, trunk hits the ground at 150, stays down.
  tilt[2, 140:] = 1.3
  contact[2, 150:] = True
  # 3: like 2 but nothing touches (contact required).
  tilt[3, 140:] = 1.3
  cfg = BuildConfig(raw=(), output_dir="")
  falls = fall_step(tilt, height, contact, 100, cfg, hold_steps=25)
  assert falls.tolist() == [t, t, 140, t]
  loose = BuildConfig(raw=(), output_dir="", fall_needs_contact=False)
  assert fall_step(tilt, height, contact, 100, loose, 25).tolist() == [t, t, 140, 140]


def test_the_online_fall_counts_a_tip_at_once_and_a_sag_once_it_stays():
  t, n = 200, 4
  tilt = torch.full((n, t), 0.1)
  height = torch.full((n, t), 0.30)
  contact = torch.zeros(n, t, dtype=torch.bool)
  # 0: stumbles over 60 deg at 120 and recovers: tipped, counted at once.
  tilt[0, 120:126] = 1.2
  # 1: trunk read 0.12 m over a stair edge at 120, no contact, walks on.
  height[1, 120:124] = 0.12
  # 2: trunk sinks onto the ground at 140 and stays: confirmed 0.5 s later.
  height[2, 140:] = 0.12
  contact[2, 140:] = True
  # 3: like 2 but back up by 150.
  height[3, 140:150] = 0.12
  contact[3, 140:150] = True
  monitor = FallMonitor(n, hold_steps=25, device="cpu")
  down = torch.stack(
    [monitor.update(tilt[:, k], height[:, k], contact[:, k]) for k in range(t)], 1
  )
  first = torch.where(down.any(1), down.float().argmax(1), t)
  assert first.tolist() == [120, t, 165, t]


def test_the_latent_head_starts_at_the_scale_of_its_target():
  # HIM's latent moves by about 2e-3; an untrained head must not start
  # hundreds of spreads off, or the latent loss swamps the rest.
  torch.manual_seed(0)
  stats = _unit_stats() | {"him_latent": {"mean": [0.01] * 16, "std": [2e-3] * 16}}
  cfg = ImpactModelCfg(d_model=32, context_layers=1, decoder_layers=1)
  model = ImpactModel(cfg, stats)
  batch = _fake_batch()
  batch["latent_future"] = 0.01 + 2e-3 * torch.randn(3, 50, 16)
  std = torch.full((16,), 2e-3)
  _, terms = impact_loss(model(batch), batch, LossWeights(), 10, latent_std=std)
  assert terms["latent"] < 10


def test_the_held_out_faults_go_to_test_fault_on_every_terrain_seed():
  joints = [
    f"{leg}_{p}_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for p in ("hip", "thigh", "calf")
  ]
  cases = [
    ("RL_calf_joint", "LOCK", True),
    ("FL_calf_joint", "LOCK", True),
    ("FR_thigh_joint", "WEAK", True),
    ("RR_hip_joint", "WEAK", True),
    ("RR_hip_joint", "LOCK", False),
    ("FR_hip_joint", "WEAK", False),
  ]
  params = np.tile(np.array(NOMINAL, np.float32), (len(cases), 12, 1))
  family = np.zeros(len(cases), np.int64)
  for i, (joint, fam, _) in enumerate(cases):
    j = joints.index(joint)
    family[i] = FaultFamily[fam]
    params[i, j] = (1.0, 1.0) if fam == "LOCK" else (0.0, 0.0)
  cfg = BuildConfig(raw=(), output_dir="")
  for seed in (2, 10, 11):  # train, test_terrain and val seeds
    split = _split(seed, family, params, joints, cfg)
    assert [s == 3 for s in split] == [held for *_, held in cases]


def test_priced_faults_draw_only_the_named_pairs():
  pairs = ("FR_hip_joint:free", "RL_thigh_joint:lock")
  params, family = sample_fault_params(400, 12, FaultMixture(pairs=pairs), "cpu")
  joint = faulted_joint(params)
  limp = params[torch.arange(400), joint, TORQUE] == 0
  locked = params[torch.arange(400), joint, LOCK] == 1
  assert set(joint.tolist()) == {JOINT_NAMES.index("FR_hip_joint"), 7}
  assert (limp == (joint == 3)).all() and (locked == (joint == 7)).all()
  assert (family[joint == 7] == FaultFamily.LOCK).all()
  assert 150 < int(limp.sum()) < 250
  # Balanced: each pair exactly as often, in a shuffled order.
  mix = FaultMixture(pairs=pairs, balanced=True)
  params, _ = sample_fault_params(64, 12, mix, "cpu")
  joint = faulted_joint(params)
  assert int((joint == 3).sum()) == 32 and not (joint[:32] == 3).all()


def test_route_progress_measures_how_far_along_however_the_robot_wandered():
  start = torch.zeros(4, 2)
  corner_then_goal = torch.tensor([[2.0, 0.0], [2.0, 3.0]]).expand(4, 2, 2)
  xy = torch.tensor(
    [
      [1.0, 0.0],  # Halfway down the first leg.
      [1.0, -0.8],  # As far, 0.8 m off to the side.
      [2.3, 1.0],  # 1 m up the second leg.
      [-1.0, 0.5],  # Behind the start: no progress.
    ]
  )
  got = route_progress(xy, start, corner_then_goal)
  assert torch.allclose(got, torch.tensor([1.0, 1.0, 3.0, 0.0]))


def test_fire_now_fires_only_faults_not_yet_fired():
  # A stand-in fault term: only the bookkeeping fire_now touches.
  term: Any = object.__new__(inject_kinodynamic_fault)
  term.faulted = torch.tensor([False, True, False])
  fired = []
  term._fire = lambda rows: fired.extend(rows.tolist())
  term.fire_now(torch.tensor([True, True, False]))
  assert fired == [0]


def test_samples_keep_what_the_driver_committed_to_and_old_shards_say_none():
  from failbench.data.build import NO_SOURCE, plan_sources

  raw = {"plan_source": np.array([0, 1, 2, 3], np.uint8)}
  assert plan_sources(raw, np.array([3, 1])).tolist() == [3, 1]
  assert plan_sources({}, np.array([0, 2])).tolist() == [NO_SOURCE] * 2


def test_a_three_second_model_trains_and_is_scored_within_its_window():
  torch.manual_seed(0)
  cfg = ImpactModelCfg(d_model=32, context_layers=1, decoder_layers=1, num_steps=30)
  model = ImpactModel(cfg, _unit_stats())
  batch = _fake_batch(t=30)
  batch["safe"] = batch["upright"].clone()
  pred = model(batch)
  assert pred["drift"].shape == (3, 30, 3)
  total, _ = impact_loss(pred, batch, LossWeights(), 10, "safe")
  total.backward()
  m = impact_metrics(pred, batch, fall_label="safe")
  assert {"drift_err_1s", "drift_err_2s", "drift_err_3s"} <= set(m)
  assert "drift_err_5s" not in m


def test_a_three_second_model_is_evaluated_per_split_and_terrain():
  torch.manual_seed(0)
  cfg = ImpactModelCfg(d_model=32, context_layers=1, decoder_layers=1, num_steps=30)
  model = ImpactModel(cfg, _unit_stats()).eval()
  b = 24  # per_terrain needs 20 samples of a terrain
  batch = _fake_batch(b, t=30)
  batch["safe"] = batch["upright"].clone()
  batch["terrain_id"] = torch.zeros(b, dtype=torch.long)
  batch["prefault_mode"] = torch.zeros(b, dtype=torch.long)
  batch["state_true_vel"] = torch.rand(b, 3)
  samples = cast(Any, [{k: v[i] for k, v in batch.items()} for i in range(b)])
  dyn = cast(Any, SimpleNamespace(model=model))
  eval_cfg = EvalConfig(checkpoint="", root="", output_dir="", num_workers=0)
  r = evaluate_split(dyn, samples, eval_cfg, "cpu", "safe")
  assert "drift_err_3s" in r and "drift_err_5s" not in r
  assert set(r["per_terrain"]["rough"]) >= {"drift_err_3s", "zero_err_3s"}
