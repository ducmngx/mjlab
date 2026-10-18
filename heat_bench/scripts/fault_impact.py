"""Headless fault impact dataset: path drift and contact risk after a fault.

Runs a trained checkpoint -- the reference Go1 policy on ``go2_eval_env_cfg``,
or a Go2 HIMLoco policy (``--task Go2-HIM-Rough[-Original]``) on its own
task -- with ``add_joint_fault_injection``: each env walks a sampled constant command on
curriculum-layout terrain, one or more joints fail at a sampled time (see
``heat_bench/envs_mjlab/joint_fault_injection.py``), and nothing resets.
Each trial is logged from ``pre_fault_s`` before the fault until
``horizon_s`` (5 s) after it -- through any fall, unless ``post_fall_s``
cuts it short. A robot that survives the fault keeps walking (and
drifting); one that falls is logged through its impact and after. No
pushes: fragility comes from the fault and the terrain alone.

Only the trunk body (its box and the head sphere, ``TRUNK_GROUPS``) counts
as contact with the environment: a fall is the trunk touching the terrain,
and contact rows, events, impact zones and the body sweep are trunk-only.
Leg contact is still logged per step (``dense_part_active``) as context.

Everything goes into one ``.npz`` (plot it with
``heat_bench/scripts/plot_fault_impact.py``):

- ``dense_*`` ``[B, T, ...]``: per-step kinematics, command, velocity error
  ``dv_b``, the planar path vs the commanded one in the fault yaw frame,
  joint state, per-body-part contact, fall indicators. Step ``j`` is
  ``dt_s[b, j]`` seconds after env ``b``'s fault (negative = before);
  ``valid`` marks logged steps.
- ``contact_*``: one row per trunk-terrain contact (slot and step).
- ``event_*``: trunk contact events (trunk box or head, onset/end time
  after the fault, order, peak force, impulse, where).
- with an HIMLoco policy, ``dense_him_vel_est`` ``[B, T, 3]`` (the
  estimator's base velocity, m/s, base frame), ``dense_him_latent``
  ``[B, T, 16]``, and ``dense_dv_est_b``: actual minus estimated velocity.
  The estimate at step ``t`` comes from the history up to ``t``, which HIM
  trains to predict the velocity after that step's action, so
  ``dv_est_b[t] = lin_vel_b[t] - him_vel_est[t - 1]`` (NaN at ``t = 0``).
- per-trial labels: fault, command, terrain tile, ``fell`` /
  ``time_to_fall_s``, first-contact times per part, peak force, final path
  deviation, elevation maps and contact risk grids at fault time
  (``*_fy``) and at fall onset (``*_fl``).

Frames, all gravity-aligned and yaw-only like the elevation map:
``fy`` = root link at fault time, ``fl`` = root link at fall onset.
``*_b`` = trunk frame at that instant. ``*_w`` = world.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import go2_him.him.tasks  # noqa: F401  (registers the Go2 HIM tasks)
import numpy as np
import torch
import tyro
from go2_him.him.env_cfg import LIN_VEL_SCALE
from go2_him.him.modules import HIMActor

from heat_bench.envs_mjlab.fault_impact import (
  BODY_GROUPS,
  FAULT_ELEVATION_SENSOR,
  RISK_CHANNELS,
  SWEEP_CHANNELS,
  TRUNK_GROUPS,
  GridSpec,
  SweepMap,
  assemble_risk,
  body_group,
  body_part,
  extract_contact_events,
  fire_step,
  geom_sample_points,
  heading,
  integrate_command_path,
  path_deviation,
  rasterize_contacts,
  rotate_to_yaw_frame,
  terrain_tile,
  tilt_angle,
  to_yaw_frame,
  window_risk,
)
from heat_bench.envs_mjlab.go2_eval_env_cfg import (
  add_box_payload,
  add_joint_fault_injection,
  go2_eval_env_cfg,
)
from heat_bench.envs_mjlab.joint_fault_injection import (
  FAULT_CONTACT_SENSOR,
  FaultMode,
  contacts_in_base_frame,
)
from heat_bench.scripts.policy_loading import REFERENCE_TASK_ID, load_runner
from heat_bench.scripts.render_fault_impact import render_trials
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.sensor import GridPatternCfg, RayCastSensor
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi
from mjlab.utils.torch import configure_torch_backends

GO2_HIM_RANDOMIZATION = ("payload", "base_com", "pd_gains", "foot_friction")

ALL_LEG_JOINTS = tuple(
  f"{leg}_{j}_joint"
  for leg in ("FR", "FL", "RR", "RL")
  for j in ("hip", "thigh", "calf")
)


@dataclass(frozen=True)
class FaultImpactConfig:
  """Configuration for the headless fault impact dataset run."""

  checkpoint_file: str | None = None
  """Path to a local checkpoint file. Mutually exclusive with wandb_run_path."""
  task: str = REFERENCE_TASK_ID
  """Task the checkpoint was trained on. The reference Go1 task runs on
  heat_bench's ``go2_eval_env_cfg``; ``Go2-HIM-Rough`` /
  ``Go2-HIM-Rough-Original`` run the Go2 HIMLoco policy on its own task
  (play variant, domain randomization off) and log HIM's estimates."""
  wandb_run_path: str | None = None
  """W&B run path ('entity/project/run_id') to resolve a checkpoint from."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run (e.g. 'model_4000.pt')."""
  num_envs: int = 256
  """Number of parallel environments (= number of trials)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  joints: tuple[str, ...] = ALL_LEG_JOINTS
  """Candidate fault joints; each env draws num_fault_joints of them."""
  num_fault_joints: int = 1
  """Joints failing at once per env."""
  modes: tuple[str, ...] = ("lock", "free", "none")
  """Fault modes; each env draws one. The study's two failure modes are
  lock (jammed joint) and free (limp joint); 'none' trials are the no-fault
  baseline. 'limit' is also available."""
  fault_time_s: tuple[float, float] = (3.0, 5.0)
  """Fault time range after the start, sampled uniformly per env."""
  limit_half_width_rad: float = 0.15
  """Half-width of the LIMIT band around the fault-time angle."""
  lin_vel_x: tuple[float, float] = (0.3, 1.0)
  """Forward command range (m/s); one constant value per trial."""
  lin_vel_y: tuple[float, float] = (-0.3, 0.3)
  """Lateral command range (m/s)."""
  ang_vel_z: tuple[float, float] = (-0.5, 0.5)
  """Yaw-rate command range (rad/s)."""
  terrain: Literal["rough", "stairs", "stairs_up", "course"] = "rough"
  """Terrain preset (curriculum layout: tile row = difficulty, col = type).
  ``stairs_up`` spawns every robot at the bottom of a stair pit, so walking
  out climbs; pair it with a straight command (``--lin-vel-y 0 0
  --ang-vel-z 0 0``) and a fault time while on the steps (~2-5 s).
  ``course`` spawns every robot at the start of a 16 m obstacle lane facing
  +x, three columns per floor type, with walls, pillars, pole fences and crates
  beside the lane (see envs_mjlab/course_terrain.py)."""
  terrain_seed: int | None = None
  """Terrain generator seed; ``None`` keeps the preset's fixed seed. Give
  each job its own seed for fresh obstacle layouts and rough ground. The
  seed used is saved as ``terrain_seed``."""
  map_size_m: float = 4.0
  """Side of the square elevation map (0.1 m cells) that contacts and risk
  grids are mapped onto. On ``course`` at up to 1 m/s, 4 m keeps ~91% of
  post-fault trunk contacts on the map, 6 m ~98.5% and 8 m ~99.9%."""
  pre_fault_s: float = 1.0
  """Also log this long before the fault, as a baseline."""
  horizon_s: float = 5.0
  """Log this long after the fault."""
  post_fall_s: float | None = None
  """Stop logging this long after a fall. None logs the full horizon, so
  every trial fills every window."""
  window_s: float = 1.0
  """Width of the per-window impact maps (``risk_window_fy``): [0, 1),
  [1, 2), ... seconds after the fault, up to ``horizon_s``."""
  tilt_fall_deg: float | None = None
  """Also count a trunk tilt beyond this (deg) as a fall. Off by default:
  a fall is the trunk touching the terrain. Tilt is logged either way."""
  min_contact_force_n: float = 1.0
  """Contacts carrying less force than this are ignored (margin touches with
  no constraint force, which would otherwise count as falls)."""
  debounce_steps: int = 2
  """Contact-free steps that don't split a contact event (bounces)."""
  payload_kg: tuple[float, float] | None = None
  """Box payload mass range (kg) strapped on the trunk. None = no payload."""
  output_file: str = "heat_bench_fault_impact.npz"
  """Where to save the dataset."""
  render_trials: int = 2
  """Render this many trials per mode (fallen ones first) as image strips:
  the robot at the fault and at each window end, with the elevation map,
  impact cells and paths drawn in (see render_fault_impact.py). 0 = off."""
  render_dir: str | None = None
  """Where to save the strips; defaults to '<output_file stem>_frames'."""


def _np(x: torch.Tensor) -> np.ndarray:
  return x.detach().cpu().numpy()


def _base_env_cfg(task: str) -> ManagerBasedRlEnvCfg:
  """The env a ``task`` checkpoint runs on, before fault injection."""
  if task == REFERENCE_TASK_ID:
    return go2_eval_env_cfg(play=True)
  cfg = load_env_cfg(task, play=True)
  # Nominal robot for clean fault labels: no randomized mass, COM, gains or
  # friction (pushes and disturbances are already off in play mode).
  for name in GO2_HIM_RANDOMIZATION:
    cfg.events.pop(name, None)
  return cfg


def _dv_est(dense: dict[str, torch.Tensor]) -> torch.Tensor:
  """Actual base velocity minus the previous step's HIM estimate (HIM
  predicts the velocity after the action taken on that history)."""
  dv = torch.full_like(dense["lin_vel_b"], float("nan"))
  dv[:, 1:] = dense["lin_vel_b"][:, 1:] - dense["him_vel_est"][:, :-1]
  both = dense["valid"][:, 1:] & dense["valid"][:, :-1]
  dv[:, 1:][~both] = float("nan")
  return dv


def run_fault_impact(cfg: FaultImpactConfig) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = _base_env_cfg(cfg.task)
  add_joint_fault_injection(
    env_cfg,
    cfg.joints,
    modes=cfg.modes,
    fault_time_s=cfg.fault_time_s,
    num_fault_joints=cfg.num_fault_joints,
    limit_half_width_rad=cfg.limit_half_width_rad,
    command_vel=None,
    command_ranges=(cfg.lin_vel_x, cfg.lin_vel_y, cfg.ang_vel_z),
    terrain=cfg.terrain,
    terrain_seed=cfg.terrain_seed,
    elevation_map=True,
    elevation_map_size=cfg.map_size_m,
  )
  if cfg.payload_kg is not None:
    add_box_payload(env_cfg, cfg.payload_kg)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped, runner = load_runner(
    env,
    device,
    cfg.checkpoint_file,
    cfg.wandb_run_path,
    cfg.wandb_checkpoint_name,
    cfg.log_root,
    cfg.task,
  )
  policy = runner.get_inference_policy(device=device)
  actor = runner.alg.get_policy()
  him = isinstance(actor, HIMActor)

  fault = env.event_manager.get_term_cfg("joint_fault_injection").func
  payload = (
    env.event_manager.get_term_cfg("box_payload").func
    if cfg.payload_kg is not None
    else None
  )
  robot = env.scene["robot"]
  sensor = env.scene[FAULT_CONTACT_SENSOR]
  elev = env.scene[FAULT_ELEVATION_SENSOR]
  assert isinstance(elev, RayCastSensor)
  pattern = elev.cfg.pattern
  assert isinstance(pattern, GridPatternCfg)
  grid = GridSpec(size=pattern.size, resolution=pattern.resolution)
  assert env.scene.terrain is not None
  gen = env.scene.terrain.cfg.terrain_generator
  assert gen is not None

  # Contact slots -> geoms -> body parts.
  geom_names = list(sensor.primary_names)
  num_slots = sensor.cfg.num_slots
  part_names = list(dict.fromkeys(body_part(g) for g in geom_names))
  part_groups = [body_group(p) for p in part_names]
  slot_geom = torch.arange(len(geom_names) * num_slots, device=device) // num_slots
  geom_part = torch.tensor(
    [part_names.index(body_part(g)) for g in geom_names], device=device
  )
  slot_part = geom_part[slot_geom]
  num_parts = len(part_names)
  part_trunk = torch.tensor([g in TRUNK_GROUPS for g in part_groups], device=device)
  slot_logged = part_trunk[slot_part]

  # Body-sweep footprint: spheres covering the robot's collision geoms.
  sweep_geoms = [g for g in geom_names if body_group(g) in TRUNK_GROUPS]
  sweep_gid = robot.indexing.geom_ids[
    [robot.geom_names.index(g) for g in sweep_geoms]
  ].long()
  gid_np = sweep_gid.cpu().numpy()
  mj_model = env.sim.mj_model
  s_geom, s_off, s_rad = geom_sample_points(
    mj_model.geom_type[gid_np], mj_model.geom_size[gid_np], mj_model.geom_rbound[gid_np]
  )
  s_gid = sweep_gid[s_geom.to(device)]
  s_off, s_rad = s_off.to(device), s_rad.to(device)

  n = cfg.num_envs
  step_dt = env.step_dt
  pre = round(cfg.pre_fault_s / step_dt)
  hor = round(cfg.horizon_s / step_dt)
  num_t = pre + hor + 1
  post_fall = round(cfg.post_fall_s / step_dt) if cfg.post_fall_s is not None else num_t
  num_j = len(fault.joint_names)
  tilt_thr = (
    float(np.deg2rad(cfg.tilt_fall_deg)) if cfg.tilt_fall_deg is not None else np.inf
  )

  def buf(*shape: int, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(n, num_t, *shape, dtype=dtype, device=device)

  dense: dict[str, torch.Tensor] = {
    "cmd": buf(3),
    "lin_vel_b": buf(3),
    "ang_vel_b": buf(3),
    "projected_gravity": buf(3),
    "base_pos_w": buf(3),
    "base_quat_w": buf(4),
    "joint_pos": buf(num_j),
    "joint_vel": buf(num_j),
    "tilt": buf(),
    "part_active": buf(num_parts, dtype=torch.bool),
    "part_force": buf(num_parts),
    "part_pos_w": buf(num_parts, 3),
    "valid": buf(dtype=torch.bool),
  }
  if him:
    dense["him_vel_est"] = buf(3)
    dense["him_latent"] = buf(actor.latent_dim)
  rows: dict[str, list[torch.Tensor]] = {
    k: [] for k in ("env_id", "step", "geom", "pos_w", "force_w", "pos_b", "force_b")
  }
  elevation_fy = torch.full((n, grid.ny, grid.nx), float("nan"), device=device)
  elevation_fl = torch.full((n, grid.ny, grid.nx), float("nan"), device=device)
  fall_rel = torch.full((n,), -1, dtype=torch.long, device=device)
  fall_cause = torch.zeros(n, dtype=torch.long, device=device)
  fall_pos_w = torch.zeros(n, 3, device=device)
  fall_quat_w = torch.zeros(n, 4, device=device)
  fall_quat_w[:, 0] = 1.0
  pre_fault_fall = torch.zeros(n, dtype=torch.bool, device=device)
  tile_fault = torch.full((n, 2), -1, dtype=torch.long, device=device)
  tile_fall = torch.full((n, 2), -1, dtype=torch.long, device=device)
  done = torch.zeros(n, dtype=torch.bool, device=device)
  reset_seen = torch.zeros(n, dtype=torch.bool, device=device)
  env_idx = torch.arange(n, device=device)
  # Robot pose (full qpos) at the fault and at each window end, to render.
  num_w = int(np.ceil(cfg.horizon_s / cfg.window_s - 1e-6))
  frame_rel = torch.tensor(
    [min(round(w * cfg.window_s / step_dt), hor) for w in range(num_w + 1)],
    device=device,
  )
  frame_qpos = torch.full(
    (n, num_w + 1, env.sim.data.qpos.shape[-1]), float("nan"), device=device
  )
  sweep_fy = SweepMap(n, grid, device)
  sweep_fl = SweepMap(n, grid, device)

  def body_points_w(e: torch.Tensor) -> torch.Tensor:
    xpos = env.sim.data.geom_xpos[e][:, s_gid]  # [R, S, 3]
    xmat = env.sim.data.geom_xmat[e][:, s_gid].view(len(e), -1, 3, 3)
    return xpos + (xmat @ s_off[None, :, :, None]).squeeze(-1)

  def elevation_now(e: torch.Tensor) -> torch.Tensor:
    data = elev.data
    z = data.hit_pos_w[e, :, 2] - robot.data.root_link_pos_w[e, 2:3]
    z = torch.where(data.distances[e] < 0, float("nan"), z)
    return z.view(-1, grid.ny, grid.nx)

  obs = wrapped.get_observations()
  prev_len = env.episode_length_buf.clone()
  max_steps = int((cfg.fault_time_s[1] + cfg.horizon_s) / step_dt) + 10
  print(f"[INFO] Running {n} fault trials for up to {max_steps} steps...")
  for _ in range(max_steps):
    with torch.no_grad():
      obs, _, _, _ = wrapped.step(policy(obs))
    # Terminations are stripped, so a shrinking episode counter means an
    # unexpected reset (and a re-sampled fault): drop that env.
    reset_seen |= env.episode_length_buf < prev_len
    prev_len = env.episode_length_buf.clone()

    # The same step the fault term fires on, so rel == 0 is the fault.
    rel = env.episode_length_buf - fire_step(fault.fault_time_s, step_dt)
    j = rel + pre
    log = ~done & (j >= 0) & (j < num_t)
    if not log.any():
      if done.all():
        break
      continue
    e = log.nonzero().squeeze(-1)
    je = j[e]
    data = robot.data

    _, force_b, pos_w, force_w, valid = contacts_in_base_frame(sensor, robot)
    pos_b = quat_apply_inverse(
      data.root_link_quat_w[:, None, :].expand(-1, pos_w.shape[1], -1),
      pos_w - data.root_link_pos_w[:, None, :],
    )
    valid = valid & (force_w.norm(dim=-1) >= cfg.min_contact_force_n)
    fmag = force_w.norm(dim=-1) * valid
    # Per body part: any contact, summed |F|, force-weighted contact point.
    p_active = torch.zeros(n, num_parts, device=device)
    p_active.index_add_(1, slot_part, valid.float())
    p_force = torch.zeros(n, num_parts, device=device)
    p_force.index_add_(1, slot_part, fmag)
    w = (fmag + 1e-6) * valid
    p_pos = torch.zeros(n, num_parts, 3, device=device)
    p_pos.index_add_(1, slot_part, pos_w * w[..., None])
    p_w = torch.zeros(n, num_parts, device=device)
    p_w.index_add_(1, slot_part, w)
    p_pos = p_pos / p_w.clamp(min=1e-12)[..., None]
    p_active = p_active > 0

    cmd = env.command_manager.get_command("twist")
    assert cmd is not None
    tilt = tilt_angle(data.projected_gravity_b)
    step_values: dict[str, torch.Tensor] = {
      "cmd": cmd,
      "lin_vel_b": data.root_link_lin_vel_b,
      "ang_vel_b": data.root_link_ang_vel_b,
      "projected_gravity": data.projected_gravity_b,
      "base_pos_w": data.root_link_pos_w,
      "base_quat_w": data.root_link_quat_w,
      "joint_pos": data.joint_pos,
      "joint_vel": data.joint_vel,
      "tilt": tilt,
      "part_active": p_active,
      "part_force": p_force,
      "part_pos_w": p_pos,
    }
    if him:
      with torch.no_grad():
        vel_est, latent = actor.estimate(obs)
      step_values["him_vel_est"] = vel_est / LIN_VEL_SCALE
      step_values["him_latent"] = latent
    for k, v in step_values.items():
      dense[k][e, je] = v[e].to(dense[k].dtype)
    dense["valid"][e, je] = True

    hit = valid & slot_logged[None, :] & log[:, None]
    if hit.any():
      he, hc = hit.nonzero(as_tuple=True)
      rows["env_id"].append(he)
      rows["step"].append(j[he])
      rows["geom"].append(slot_geom[hc])
      rows["pos_w"].append(pos_w[he, hc])
      rows["force_w"].append(force_w[he, hc])
      rows["pos_b"].append(pos_b[he, hc])
      rows["force_b"].append(force_b[he, hc])

    # Pose snapshots for rendering.
    snap = log[:, None] & (rel[:, None] == frame_rel[None, :])
    if snap.any():
      se, sk = snap.nonzero(as_tuple=True)
      frame_qpos[se, sk] = env.sim.data.qpos[se].float()

    # Elevation map at the fault.
    at_fault = log & (rel == 0)
    if at_fault.any():
      ef = at_fault.nonzero().squeeze(-1)
      elevation_fy[ef] = elevation_now(ef)
      r, c = terrain_tile(data.root_link_pos_w[ef], gen)
      tile_fault[ef] = torch.stack([r, c], dim=-1)

    # Fall onset: first post-fault step with the trunk on the terrain (or,
    # if enabled, a big tilt).
    trunk_hit = (p_active & part_trunk).any(dim=-1)
    tilted = tilt > tilt_thr
    falling = log & (trunk_hit | tilted)
    pre_fault_fall |= falling & (rel < 0)
    new_fall = falling & (rel >= 0) & (fall_rel < 0)
    if new_fall.any():
      nf = new_fall.nonzero().squeeze(-1)
      fall_rel[nf] = rel[nf]
      fall_cause[nf] = trunk_hit[nf].long() + 2 * tilted[nf].long()
      fall_pos_w[nf] = data.root_link_pos_w[nf]
      fall_quat_w[nf] = data.root_link_quat_w[nf]
      elevation_fl[nf] = elevation_now(nf)
      r, c = terrain_tile(data.root_link_pos_w[nf], gen)
      tile_fall[nf] = torch.stack([r, c], dim=-1)

    # Body sweep after the fault, and after the fall onset.
    sw = log & (rel >= 0) & fault.faulted
    if sw.any():
      es = sw.nonzero().squeeze(-1)
      pts_w = body_points_w(es)
      dt_now = rel[es].float() * step_dt
      pts = to_yaw_frame(pts_w, fault.fault_base_pos_w[es], fault.fault_base_quat_w[es])
      sweep_fy.update(es, pts, s_rad, elevation_fy[es], dt_now)
      after = (fall_rel[es] >= 0) & (rel[es] >= fall_rel[es])
      if after.any():
        ea = es[after]
        pts = to_yaw_frame(pts_w[after], fall_pos_w[ea], fall_quat_w[ea])
        sweep_fl.update(ea, pts, s_rad, elevation_fl[ea], dt_now[after])

    fell = fall_rel >= 0
    done |= log & ((rel >= hor) | (fell & (rel >= fall_rel + post_fall)))
    if done.all():
      break

  if not done.all():
    print(f"[WARN] {int((~done).sum())} envs didn't finish their window.")
  mismatch = fault.faulted & (
    fault.fault_step != fire_step(fault.fault_time_s, step_dt)
  )
  if mismatch.any():
    print(f"[WARN] {int(mismatch.sum())} envs faulted off the predicted step.")
  keep = done & ~reset_seen & ~pre_fault_fall & ~mismatch
  if reset_seen.any():
    print(f"[WARN] Dropping {int(reset_seen.sum())} envs that reset mid-run.")
  if pre_fault_fall.any():
    print(f"[WARN] Dropping {int(pre_fault_fall.sum())} envs that fell pre-fault.")

  # Spawn point of each env's fault-time tile (a stair pit's centre, say).
  terrain_origins = env.scene.terrain.terrain_origins
  assert terrain_origins is not None
  tile_origin_fault = torch.where(
    (tile_fault >= 0).all(dim=-1, keepdim=True),
    terrain_origins[tile_fault[:, 0].clamp(min=0), tile_fault[:, 1].clamp(min=0)],
    float("nan"),
  )

  # Frames: fault yaw (fy) and fall-onset yaw (fl).
  f_pos, f_quat = fault.fault_base_pos_w, fault.fault_base_quat_w
  fell = fall_rel >= 0
  valid_t = dense["valid"]
  rel_t = torch.arange(num_t, device=device) - pre
  dt_s = torch.where(valid_t, rel_t[None, :] * step_dt, float("nan"))

  base_pos_fy = to_yaw_frame(dense["base_pos_w"], f_pos, f_quat)
  yaw_fy = wrap_to_pi(heading(dense["base_quat_w"]) - heading(f_quat)[:, None])
  actual_fy = torch.cat([base_pos_fy[..., :2], yaw_fy[..., None]], dim=-1)
  expected_fy = torch.full((n, num_t, 3), float("nan"), device=device)
  expected_fy[:, pre] = 0.0
  expected_fy[:, pre + 1 :] = integrate_command_path(
    dense["cmd"][:, pre + 1 :], step_dt
  )
  post_valid = valid_t & (rel_t >= 0)[None, :]
  expected_fy[~post_valid] = float("nan")
  deviation_fy = path_deviation(actual_fy, expected_fy)
  dv_b = torch.cat(
    [
      dense["lin_vel_b"][..., :2] - dense["cmd"][..., :2],
      dense["ang_vel_b"][..., 2:] - dense["cmd"][..., 2:],
    ],
    dim=-1,
  )
  dv_b[~valid_t] = float("nan")
  part_pos_fy = to_yaw_frame(dense["part_pos_w"], f_pos, f_quat)
  part_pos_fy[~dense["part_active"]] = float("nan")
  last = (valid_t.long() * torch.arange(num_t, device=device)).argmax(dim=-1)
  final_deviation = deviation_fy[env_idx, last]

  # Sparse contact rows.
  contacts = {
    k: torch.cat(v) if v else torch.empty(0, device=device) for k, v in rows.items()
  }
  if rows["env_id"]:
    c_env = contacts["env_id"]
    contacts["dt"] = (contacts["step"] - pre).float() * step_dt
    contacts["part"] = geom_part[contacts["geom"]]
    contacts["pos_fy"] = to_yaw_frame(
      contacts["pos_w"][:, None], f_pos[c_env], f_quat[c_env]
    )[:, 0]
    contacts["force_fy"] = rotate_to_yaw_frame(
      contacts["force_w"][:, None], f_quat[c_env]
    )[:, 0]
    pos_fl = to_yaw_frame(
      contacts["pos_w"][:, None], fall_pos_w[c_env], fall_quat_w[c_env]
    )[:, 0]
    after_fall = fell[c_env] & (contacts["step"] - pre >= fall_rel[c_env])
    contacts["pos_fl"] = torch.where(after_fall[:, None], pos_fl, float("nan"))
    fmag_c = contacts["force_w"].norm(dim=-1)
    post = contacts["dt"] >= 0
    risk_fy, outside_fy = rasterize_contacts(
      c_env[post], contacts["pos_fy"][post], fmag_c[post], contacts["dt"][post], n, grid
    )
    risk_fl, outside_fl = rasterize_contacts(
      c_env[after_fall],
      contacts["pos_fl"][after_fall],
      fmag_c[after_fall],
      contacts["dt"][after_fall],
      n,
      grid,
    )
    peak_force = torch.zeros(n, device=device)
    peak_force.scatter_reduce_(0, c_env[post], fmag_c[post], reduce="amax")
  else:
    nan_grid = torch.full((n, 3, grid.ny, grid.nx), float("nan"), device=device)
    nan_grid[:, 0] = 0.0
    risk_fy, risk_fl = nan_grid, nan_grid.clone()
    outside_fy = outside_fl = torch.zeros(n, dtype=torch.long, device=device)
    peak_force = torch.zeros(n, device=device)

  # Contact events per body part.
  ev_active = dense["part_active"] & part_trunk
  events = extract_contact_events(
    _np(ev_active), _np(dense["part_force"]), _np(dt_s), step_dt, cfg.debounce_steps
  )
  ev_env = torch.from_numpy(events["env_id"]).to(device)
  ev_on = torch.from_numpy(events["onset_step"]).to(device)
  ev_part = torch.from_numpy(events["part"]).to(device)
  onset_w = dense["part_pos_w"][ev_env, ev_on, ev_part]
  events["pos_fy"] = _np(
    to_yaw_frame(onset_w[:, None], f_pos[ev_env], f_quat[ev_env])[:, 0]
  )
  events["pos_b"] = _np(
    quat_apply_inverse(
      dense["base_quat_w"][ev_env, ev_on], onset_w - dense["base_pos_w"][ev_env, ev_on]
    )
  )
  ev_fl = fell[ev_env] & (ev_on - pre >= fall_rel[ev_env])
  ev_pos_fl = to_yaw_frame(onset_w[:, None], fall_pos_w[ev_env], fall_quat_w[ev_env])[
    :, 0
  ]
  ev_pos_fl[~ev_fl] = float("nan")
  events["pos_fl"] = _np(ev_pos_fl)
  ev_dt = torch.from_numpy(events["onset_dt"]).to(device)
  ev_force = torch.from_numpy(events["peak_force_n"]).to(device)
  ev_post = ev_dt >= 0
  impacts_fy, _ = rasterize_contacts(
    ev_env[ev_post],
    torch.from_numpy(events["pos_fy"]).to(device)[ev_post],
    ev_force[ev_post],
    ev_dt[ev_post],
    n,
    grid,
  )
  impacts_fl, _ = rasterize_contacts(
    ev_env[ev_fl], ev_pos_fl[ev_fl], ev_force[ev_fl], ev_dt[ev_fl], n, grid
  )
  risk_fy = assemble_risk(risk_fy, impacts_fy[:, 0])
  risk_fl = assemble_risk(risk_fl, impacts_fl[:, 0])

  # Impact maps per time window after the fault, in the frozen fault frame,
  # plus the trunk pose (x, y, z, heading) at each window's end.
  edges = torch.arange(num_w + 1, device=device) * cfg.window_s
  edges[-1] = min(float(edges[-1]), cfg.horizon_s) + 1e-6  # Include dt == horizon.
  if rows["env_id"]:
    c_win = (
      contacts["env_id"],
      contacts["pos_fy"],
      contacts["force_w"].norm(dim=-1),
      contacts["dt"],
    )
  else:
    empty = torch.empty(0, device=device)
    c_win = (empty.long(), empty.view(0, 3), empty, empty)
  e_win = (
    ev_env,
    torch.from_numpy(events["pos_fy"]).to(device),
    ev_force,
    ev_dt,
  )
  risk_window_fy = window_risk(c_win, e_win, edges, n, grid)
  end_j = (pre + torch.round(edges[1:] / step_dt).long()).clamp(max=num_t - 1)
  pose_window_fy = torch.cat([base_pos_fy[:, end_j], yaw_fy[:, end_j, None]], dim=-1)
  pose_window_fy[~valid_t[:, end_j]] = float("nan")
  events["group"] = np.array(
    [BODY_GROUPS.index(part_groups[p]) for p in events["part"]], dtype=np.int64
  )
  # First post-fault contact time and event count, per part and per group.
  first_part_dt = np.full((n, num_parts), np.nan, dtype=np.float32)
  num_part_events = np.zeros((n, num_parts), dtype=np.int64)
  post_ev = events["onset_dt"] >= 0
  at = (events["env_id"][post_ev], events["part"][post_ev])
  np.fmin.at(first_part_dt, at, events["onset_dt"][post_ev])  # fmin skips NaN.
  np.add.at(num_part_events, at, 1)
  group_idx = np.array([BODY_GROUPS.index(g) for g in part_groups])
  first_group_dt = np.full((n, len(BODY_GROUPS)), np.nan, dtype=np.float32)
  for g in range(len(BODY_GROUPS)):
    if (group_idx == g).any():
      cols = first_part_dt[:, group_idx == g]
      first_group_dt[:, g] = np.where(
        np.isnan(cols).all(axis=1),
        np.nan,
        np.nanmin(np.nan_to_num(cols, nan=np.inf), axis=1),
      )

  sub_names = list(gen.sub_terrains)
  lo, hi = gen.difficulty_range
  row_difficulty = lo + (hi - lo) * np.arange(gen.num_rows) / max(gen.num_rows - 1, 1)
  out: dict[str, np.ndarray] = {
    **{f"dense_{k}": _np(v) for k, v in dense.items() if k != "part_pos_w"},
    "dense_dt_s": _np(dt_s),
    "dense_dv_b": _np(dv_b),
    **({"dense_dv_est_b": _np(_dv_est(dense))} if him else {}),
    "dense_base_pos_fy": _np(base_pos_fy),
    "dense_actual_fy": _np(actual_fy),
    "dense_expected_fy": _np(expected_fy),
    "dense_deviation_fy": _np(deviation_fy),
    "dense_part_pos_fy": _np(part_pos_fy),
    **{f"contact_{k}": _np(v) for k, v in contacts.items()},
    **{f"event_{k}": v for k, v in events.items()},
    "geom_names": np.array(geom_names),
    "part_names": np.array(part_names),
    "part_groups": np.array(part_groups),
    "group_names": np.array(BODY_GROUPS),
    "joint_names": np.array(fault.joint_names),
    "mode_names": np.array([m.name.lower() for m in FaultMode]),
    "terrain_names": np.array(sub_names),
    "terrain_seed": np.array(-1 if gen.seed is None else gen.seed),
    "terrain_row_difficulty": row_difficulty.astype(np.float32),
    "env_kept": _np(keep),
    "fault_joint": _np(fault.fault_joint),
    "fault_mode": _np(fault.fault_mode),
    "fault_time_s": _np(fault.fault_time_s),
    "fault_qpos": _np(fault.fault_qpos),
    "fault_base_pos_w": _np(f_pos),
    "fault_base_quat_w": _np(f_quat),
    "command": _np(dense["cmd"][:, pre]),
    "terrain_tile_fault": _np(tile_fault),
    "terrain_tile_fall": _np(tile_fall),
    "fell": _np(fell),
    "time_to_fall_s": _np(torch.where(fell, fall_rel.float() * step_dt, float("nan"))),
    "fall_cause": _np(fall_cause),
    "fall_base_pos_w": _np(fall_pos_w),
    "fall_base_quat_w": _np(fall_quat_w),
    "first_contact_dt_part": first_part_dt,
    "num_contact_events_part": num_part_events,
    "first_contact_dt_group": first_group_dt,
    "peak_force_n": _np(peak_force),
    "final_deviation_fy": _np(final_deviation),
    "elevation_fy": _np(elevation_fy),
    "elevation_fl": _np(elevation_fl),
    "risk_fy": _np(risk_fy),
    "risk_fl": _np(risk_fl),
    "risk_outside_fy": _np(outside_fy),
    "risk_outside_fl": _np(outside_fl),
    "risk_channels": np.array(RISK_CHANNELS),
    "sweep_fy": _np(sweep_fy.result()),
    "sweep_fl": _np(sweep_fl.result()),
    "sweep_channels": np.array(SWEEP_CHANNELS),
    "window_edges_s": _np(edges.clamp(max=cfg.horizon_s)),
    "risk_window_fy": _np(risk_window_fy),
    "pose_window_end_fy": _np(pose_window_fy),
    "frame_qpos": _np(frame_qpos),
    "sweep_geoms": np.array(sweep_geoms),
    # From the tile under the robot, not env.scene.env_origins: play mode's
    # randomize_terrain reset event moves env_origins to a new tile after
    # the robot has already been placed.
    "fault_tile_origin_w": _np(tile_origin_fault),
    "fault_offset_from_tile_origin": _np(f_pos - tile_origin_fault),
    "grid_size": np.array(grid.size, dtype=np.float32),
    "grid_resolution": np.array(grid.resolution, dtype=np.float32),
    "payload_mass_kg": (
      _np(payload.mass_kg) if payload is not None else np.zeros(n, dtype=np.float32)
    ),
    "step_dt": np.array(step_dt),
    "pre_fault_s": np.array(cfg.pre_fault_s),
    "horizon_s": np.array(cfg.horizon_s),
    "post_fall_s": np.array(cfg.post_fall_s if cfg.post_fall_s is not None else np.nan),
    "window_s": np.array(cfg.window_s),
    "tilt_fall_deg": np.array(
      cfg.tilt_fall_deg if cfg.tilt_fall_deg is not None else np.nan
    ),
    "min_contact_force_n": np.array(cfg.min_contact_force_n),
  }
  output_path = Path(cfg.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, allow_pickle=False, **out)

  kept = out["env_kept"]
  for mode in FaultMode:
    sel = kept & (out["fault_mode"] == mode)
    if not sel.any():
      continue
    ttf = out["time_to_fall_s"][sel]
    dev = np.linalg.norm(out["final_deviation_fy"][sel, :2], axis=-1)
    print(
      f"[INFO] {mode.name.lower():>5}: {sel.sum()} envs, "
      f"fell {out['fell'][sel].mean():.0%} "
      f"(median {np.nanmedian(ttf) if np.isfinite(ttf).any() else float('nan'):.2f} s after), "
      f"median final drift {np.nanmedian(dev):.2f} m, "
      f"median peak force {np.median(out['peak_force_n'][sel]):.0f} N"
    )
  print(f"[INFO] {len(events['env_id'])} contact events saved to {output_path}")
  if cfg.render_trials > 0:
    render_dir = Path(
      cfg.render_dir or output_path.with_name(f"{output_path.stem}_frames")
    )
    strips = render_trials(env.sim.mj_model, out, cfg.render_trials, render_dir)
    print(f"[INFO] Rendered {len(strips)} trial strips to {render_dir}")
  env.close()
  return output_path


def main():
  cfg = tyro.cli(FaultImpactConfig)
  if cfg.checkpoint_file is None and cfg.wandb_run_path is None:
    print("Must pass --checkpoint-file or --wandb-run-path.")
    sys.exit(1)
  run_fault_impact(cfg)


if __name__ == "__main__":
  main()
