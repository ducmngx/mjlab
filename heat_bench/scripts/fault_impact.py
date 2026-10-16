"""Headless fault impact dataset: path drift and contact risk after a fault.

Runs a trained checkpoint on ``go2_eval_env_cfg`` with
``add_joint_fault_injection``: each env walks a sampled constant command on
curriculum-layout terrain, one or more joints fail at a sampled time (see
``heat_bench/envs_mjlab/joint_fault_injection.py``), and nothing resets.
Each trial is logged from ``pre_fault_s`` before the fault until
``horizon_s`` after it, or until ``post_fall_s`` after a fall, whichever
comes first. A robot that survives the fault keeps walking (and drifting)
for the full horizon; one that falls is logged through its impact. No
pushes: fragility comes from the fault and the terrain alone.

Everything goes into one ``.npz`` (plot it with
``heat_bench/scripts/plot_fault_impact.py``):

- ``dense_*`` ``[B, T, ...]``: per-step kinematics, command, velocity error
  ``dv_b``, the planar path vs the commanded one in the fault yaw frame,
  joint state, per-body-part contact, fall indicators. Step ``j`` is
  ``dt_s[b, j]`` seconds after env ``b``'s fault (negative = before);
  ``valid`` marks logged steps.
- ``contact_*``: one row per robot-terrain contact (slot and step),
  feet excluded unless ``log_foot_contacts``.
- ``event_*``: per-body-part contact events (which part, onset/end time
  after the fault, order, peak force, impulse, where).
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

import numpy as np
import torch
import tyro

from heat_bench.envs_mjlab.fault_impact import (
  BODY_GROUPS,
  FAULT_ELEVATION_SENSOR,
  SEVERE_GROUPS,
  GridSpec,
  body_group,
  body_part,
  extract_contact_events,
  fire_step,
  heading,
  integrate_command_path,
  path_deviation,
  rasterize_contacts,
  rotate_to_yaw_frame,
  terrain_tile,
  tilt_angle,
  to_yaw_frame,
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
from heat_bench.scripts.policy_loading import load_policy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import GridPatternCfg, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi
from mjlab.utils.torch import configure_torch_backends

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
  modes: tuple[str, ...] = ("limit", "lock", "free", "none")
  """Fault modes; each env draws one. 'none' trials are no-fault controls."""
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
  terrain: Literal["rough", "stairs"] = "rough"
  """Terrain preset (curriculum layout: tile row = difficulty, col = type)."""
  pre_fault_s: float = 1.0
  """Also log this long before the fault, as a baseline."""
  horizon_s: float = 15.0
  """Log at most this long after the fault."""
  post_fall_s: float = 2.0
  """After a fall, keep logging this long to capture the impact, then stop."""
  tilt_fall_deg: float = 60.0
  """Trunk tilt from upright that counts as a fall."""
  severe_groups: tuple[str, ...] = SEVERE_GROUPS
  """Body groups whose terrain contact counts as a fall."""
  min_contact_force_n: float = 1.0
  """Contacts carrying less force than this are ignored (margin touches with
  no constraint force, which would otherwise count as falls)."""
  debounce_steps: int = 2
  """Contact-free steps that don't split a contact event (bounces)."""
  log_foot_contacts: bool = False
  """Also log foot contacts as rows and events (one event per footstep)."""
  payload_kg: tuple[float, float] | None = None
  """Box payload mass range (kg) strapped on the trunk. None = no payload."""
  output_file: str = "heat_bench_fault_impact.npz"
  """Where to save the dataset."""


def _np(x: torch.Tensor) -> np.ndarray:
  return x.detach().cpu().numpy()


def run_fault_impact(cfg: FaultImpactConfig) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = go2_eval_env_cfg(play=True)
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
    elevation_map=True,
  )
  if cfg.payload_kg is not None:
    add_box_payload(env_cfg, cfg.payload_kg)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped, policy = load_policy(
    env,
    device,
    cfg.checkpoint_file,
    cfg.wandb_run_path,
    cfg.wandb_checkpoint_name,
    cfg.log_root,
  )

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
  part_is_foot = torch.tensor([g == "foot" for g in part_groups], device=device)
  part_severe = torch.tensor(
    [g in cfg.severe_groups for g in part_groups], device=device
  )
  slot_logged = ~part_is_foot[slot_part] | cfg.log_foot_contacts

  n = cfg.num_envs
  step_dt = env.step_dt
  pre = round(cfg.pre_fault_s / step_dt)
  hor = round(cfg.horizon_s / step_dt)
  post_fall = round(cfg.post_fall_s / step_dt)
  num_t = pre + hor + 1
  num_j = len(fault.joint_names)
  tilt_thr = float(np.deg2rad(cfg.tilt_fall_deg))

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

    # Elevation map at the fault.
    at_fault = log & (rel == 0)
    if at_fault.any():
      ef = at_fault.nonzero().squeeze(-1)
      elevation_fy[ef] = elevation_now(ef)
      r, c = terrain_tile(data.root_link_pos_w[ef], gen)
      tile_fault[ef] = torch.stack([r, c], dim=-1)

    # Fall onset: first post-fault step with a severe contact or a big tilt.
    severe = (p_active & part_severe).any(dim=-1)
    tilted = tilt > tilt_thr
    falling = log & (severe | tilted)
    pre_fault_fall |= falling & (rel < 0)
    new_fall = falling & (rel >= 0) & (fall_rel < 0)
    if new_fall.any():
      nf = new_fall.nonzero().squeeze(-1)
      fall_rel[nf] = rel[nf]
      fall_cause[nf] = severe[nf].long() + 2 * tilted[nf].long()
      fall_pos_w[nf] = data.root_link_pos_w[nf]
      fall_quat_w[nf] = data.root_link_quat_w[nf]
      elevation_fl[nf] = elevation_now(nf)
      r, c = terrain_tile(data.root_link_pos_w[nf], gen)
      tile_fall[nf] = torch.stack([r, c], dim=-1)

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
  ev_active = dense["part_active"] & (~part_is_foot | cfg.log_foot_contacts)
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
    "grid_size": np.array(grid.size, dtype=np.float32),
    "grid_resolution": np.array(grid.resolution, dtype=np.float32),
    "payload_mass_kg": (
      _np(payload.mass_kg) if payload is not None else np.zeros(n, dtype=np.float32)
    ),
    "step_dt": np.array(step_dt),
    "pre_fault_s": np.array(cfg.pre_fault_s),
    "horizon_s": np.array(cfg.horizon_s),
    "post_fall_s": np.array(cfg.post_fall_s),
    "tilt_fall_deg": np.array(cfg.tilt_fall_deg),
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
