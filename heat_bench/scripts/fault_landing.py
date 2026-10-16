"""Headless joint-fault landing demo: walk, fail joints, record contacts.

Runs a trained checkpoint on ``go2_eval_env_cfg`` with
``add_joint_fault_injection``: every env walks at a fixed command, one or
more joints fail mid-run (LIMIT / LOCK / FREE, see
``heat_bench/envs_mjlab/joint_fault_injection.py``), and nothing resets.
From ``pre_fault_s`` before the fault to ``window_s`` after it, every
robot-terrain contact is logged with its position and the force on the
robot, in the trunk frame (and world / fault-time frames). Saved as one
``.npz``; plot it with ``heat_bench/scripts/plot_fault_landing.py``.

Frames in the output:

- ``*_b``: the trunk frame at that instant (origin at the root link).
- ``*_f``: the trunk frame frozen at fault time -- where things land
  relative to where the robot was when the joint failed.
- ``*_w``: world.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from heat_bench.envs_mjlab.fault_impact import body_group
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
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class FaultLandingConfig:
  """Configuration for the headless joint-fault landing run."""

  checkpoint_file: str | None = None
  """Path to a local checkpoint file. Mutually exclusive with wandb_run_path."""
  wandb_run_path: str | None = None
  """W&B run path ('entity/project/run_id') to resolve a checkpoint from."""
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run (e.g. 'model_4000.pt')."""
  num_envs: int = 64
  """Number of parallel environments (= number of fault trials)."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  joints: tuple[str, ...] = ("FR_calf_joint", "FL_calf_joint", "RR_calf_joint")
  """Candidate fault joints; each env draws num_fault_joints of them."""
  num_fault_joints: int = 1
  """Joints failing at once per env."""
  modes: tuple[str, ...] = ("limit", "lock", "free")
  """Fault modes; each env draws one. Add 'none' for no-fault control trials."""
  fault_time_s: tuple[float, float] = (3.0, 5.0)
  """Fault time range after the start, sampled uniformly per env."""
  limit_half_width_rad: float = 0.15
  """Half-width of the LIMIT band around the fault-time angle."""
  command_vel: tuple[float, float, float] = (1.0, 0.0, 0.0)
  """Fixed (lin_vel_x, lin_vel_y, ang_vel_z) command."""
  pre_fault_s: float = 0.5
  """Also log contacts this long before the fault, as a baseline."""
  window_s: float = 3.0
  """How long to log contacts after the fault."""
  payload_kg: tuple[float, float] | None = None
  """Box payload strapped on top of the trunk, mass sampled per env in this
  range (kg); see heat_bench/envs_mjlab/payload.py. None = no payload."""
  output_file: str = "heat_bench_fault_landing.npz"
  """Where to save the contact log."""


def run_fault_landing(cfg: FaultLandingConfig) -> Path:
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
    command_vel=cfg.command_vel,
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
  sensor = env.scene[FAULT_CONTACT_SENSOR]
  robot = env.scene["robot"]
  geom_names = sensor.primary_names
  num_slots = sensor.cfg.num_slots
  slot_geom = torch.arange(len(geom_names) * num_slots, device=device) // num_slots
  nonfoot_geom = torch.tensor(
    [body_group(n) != "foot" for n in geom_names], device=device
  )
  trunk_geom = torch.tensor(
    [body_group(n) in ("trunk", "head") for n in geom_names], device=device
  )

  n = cfg.num_envs
  step_dt = env.step_dt
  rows: dict[str, list[torch.Tensor]] = {
    k: []
    for k in (
      "env_id",
      "t",
      "geom",
      "pos_b",
      "force_b",
      "pos_f",
      "force_f",
      "pos_w",
      "force_w",
    )
  }
  first_nonfoot_t = torch.full((n,), float("nan"), device=device)
  first_nonfoot_geom = torch.full((n,), -1, dtype=torch.long, device=device)
  first_nonfoot_pos_b = torch.full((n, 3), float("nan"), device=device)
  peak_force = torch.zeros(n, device=device)
  trunk_touched = torch.zeros(n, dtype=torch.bool, device=device)
  end_base_pos_w = torch.zeros(n, 3, device=device)
  end_base_quat_w = torch.zeros(n, 4, device=device)
  done = torch.zeros(n, dtype=torch.bool, device=device)
  reset_seen = torch.zeros(n, dtype=torch.bool, device=device)

  obs = wrapped.get_observations()
  prev_len = env.episode_length_buf.clone()
  max_steps = int((cfg.fault_time_s[1] + cfg.window_s) / step_dt) + 10
  print(f"[INFO] Running {n} fault trials for up to {max_steps} steps...")
  for _ in range(max_steps):
    with torch.no_grad():
      obs, _, _, _ = wrapped.step(policy(obs))
    # Terminations are stripped, so a shrinking episode counter means an
    # unexpected reset (and a re-sampled fault): drop that env.
    reset_seen |= env.episode_length_buf < prev_len
    prev_len = env.episode_length_buf.clone()

    t = env.episode_length_buf.float() * step_dt - fault.fault_time_s
    logging = ~done & (t >= -cfg.pre_fault_s)
    pos_b, force_b, pos_w, force_w, valid = contacts_in_base_frame(sensor, robot)
    hit = valid & logging[:, None]
    if hit.any():
      e, c = hit.nonzero(as_tuple=True)
      fault_pos = fault.fault_base_pos_w[e]
      fault_quat = fault.fault_base_quat_w[e]
      rows["env_id"].append(e)
      rows["t"].append(t[e])
      rows["geom"].append(slot_geom[c])
      rows["pos_b"].append(pos_b[e, c])
      rows["force_b"].append(force_b[e, c])
      rows["pos_f"].append(quat_apply_inverse(fault_quat, pos_w[e, c] - fault_pos))
      rows["force_f"].append(quat_apply_inverse(fault_quat, force_w[e, c]))
      rows["pos_w"].append(pos_w[e, c])
      rows["force_w"].append(force_w[e, c])

    # Post-fault summaries.
    post = ~done & fault.faulted & (t >= 0)
    post_hit = valid & post[:, None]
    force_mag = force_w.norm(dim=-1) * post_hit
    peak_force = torch.maximum(peak_force, force_mag.max(dim=-1).values)
    trunk_touched |= (post_hit & trunk_geom[slot_geom]).any(dim=-1)
    nonfoot_hit = post_hit & nonfoot_geom[slot_geom]
    new_first = nonfoot_hit.any(dim=-1) & first_nonfoot_t.isnan()
    if new_first.any():
      # The strongest non-foot contact of the first step any occurred.
      slot = (force_mag * nonfoot_hit).argmax(dim=-1)
      env_idx = torch.arange(n, device=device)
      first_nonfoot_t = torch.where(new_first, t, first_nonfoot_t)
      first_nonfoot_geom = torch.where(new_first, slot_geom[slot], first_nonfoot_geom)
      first_nonfoot_pos_b = torch.where(
        new_first[:, None], pos_b[env_idx, slot], first_nonfoot_pos_b
      )

    finishing = ~done & fault.faulted & (t >= cfg.window_s)
    end_base_pos_w[finishing] = robot.data.root_link_pos_w[finishing]
    end_base_quat_w[finishing] = robot.data.root_link_quat_w[finishing]
    done |= finishing
    if done.all():
      break

  if not done.all():
    print(f"[WARN] {int((~done).sum())} envs didn't finish their window.")
  keep = done & ~reset_seen
  if reset_seen.any():
    print(f"[WARN] Dropping {int(reset_seen.sum())} envs that reset mid-run.")

  contacts: dict[str, np.ndarray] = {}
  if rows["env_id"]:
    contact_keep = keep[torch.cat(rows["env_id"])]
    contacts = {k: torch.cat(v)[contact_keep].cpu().numpy() for k, v in rows.items()}

  end_pos_f = quat_apply_inverse(
    fault.fault_base_quat_w, end_base_pos_w - fault.fault_base_pos_w
  )
  joint_names = np.array(fault.joint_names)
  out: dict[str, np.ndarray] = {
    **{f"contact_{k}": v for k, v in contacts.items()},
    "geom_names": np.array(geom_names),
    "geom_groups": np.array([body_group(g) for g in geom_names]),
    "joint_names": joint_names,
    "mode_names": np.array([m.name.lower() for m in FaultMode]),
    "env_kept": keep.cpu().numpy(),
    "fault_joint": fault.fault_joint.cpu().numpy(),
    "fault_mode": fault.fault_mode.cpu().numpy(),
    "fault_time_s": fault.fault_time_s.cpu().numpy(),
    "fault_qpos": fault.fault_qpos.cpu().numpy(),
    "fault_base_pos_w": fault.fault_base_pos_w.cpu().numpy(),
    "fault_base_quat_w": fault.fault_base_quat_w.cpu().numpy(),
    "end_base_pos_w": end_base_pos_w.cpu().numpy(),
    "end_base_quat_w": end_base_quat_w.cpu().numpy(),
    "end_base_pos_f": end_pos_f.cpu().numpy(),
    "first_nonfoot_t": first_nonfoot_t.cpu().numpy(),
    "first_nonfoot_geom": first_nonfoot_geom.cpu().numpy(),
    "first_nonfoot_pos_b": first_nonfoot_pos_b.cpu().numpy(),
    "peak_force_n": peak_force.cpu().numpy(),
    "trunk_touched": trunk_touched.cpu().numpy(),
    "payload_mass_kg": (
      payload.mass_kg.cpu().numpy() if payload is not None else np.zeros(n)
    ),
    "payload_offset": (
      payload.offset.cpu().numpy() if payload is not None else np.zeros((n, 3))
    ),
    "pre_fault_s": np.array(cfg.pre_fault_s),
    "window_s": np.array(cfg.window_s),
  }
  output_path = Path(cfg.output_file)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, allow_pickle=False, **out)

  kept = keep.cpu().numpy()
  modes = out["fault_mode"]
  for mode in FaultMode:
    sel = kept & (modes == mode)
    if sel.any():
      print(
        f"[INFO] {mode.name.lower():>5}: {sel.sum()} envs, "
        f"trunk touched {out['trunk_touched'][sel].mean():.0%}, "
        f"non-foot contact {(~np.isnan(out['first_nonfoot_t'][sel])).mean():.0%}, "
        f"median peak force {np.median(out['peak_force_n'][sel]):.0f} N"
      )
  num_contacts = len(contacts["env_id"]) if contacts else 0
  print(f"[INFO] {num_contacts} contacts saved to {output_path}")
  env.close()
  return output_path


def main():
  cfg = tyro.cli(FaultLandingConfig)
  if cfg.checkpoint_file is None and cfg.wandb_run_path is None:
    print("Must pass --checkpoint-file or --wandb-run-path.")
    sys.exit(1)
  run_fault_landing(cfg)


if __name__ == "__main__":
  main()
