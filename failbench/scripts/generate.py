"""Simulate FailBench fault trials into raw shards.

Each env walks its pre-fault command on a heat_bench fault terrain, its
joints take a sampled kinodynamic fault (``failbench/faults.py``) at a
sampled time, and its command then follows the post-fault knots
(``failbench/commands.py``). Nothing resets. Every trial is logged from
``pre_fault_s`` before the fault to ``horizon_s`` after it, at the 50 Hz
control rate, and the clean elevation map is snapshotted at each of
``snapshot_s`` after the fault. ``--rounds`` repeats the batch with fresh
faults, commands and spawn poses, one shard per round.

A shard (``<output_dir>/<prefix>_r<round>.npz``) holds, per trial ``b``:

- ``dense_*`` ``[B, T, ...]``: command, base velocity (true, and HIM's
  estimate), angular velocity, projected gravity, base pose (world), joint
  state, HIM latent, tilt, trunk contact, ``valid``. Step ``j`` is
  ``(j - pre_steps) * step_dt`` seconds after the fault.
- ``map`` ``[B, S, 64, 64]`` (fp16): terrain height relative to the trunk
  at each snapshot, NaN where a ray missed; trunk-centred and yaw-aligned
  (``fault_elevation_sensor_cfg``).
- ``fault_params`` ``[B, 12, 6]``, ``fault_family``, ``fault_time_s``;
  ``cmd_knots`` ``[B, K + 1, 3]``, ``cmd_style``; terrain tile at the fault.
- ``contact_*``: one row per non-foot robot-terrain contact (slot and step)
  at or above ``min_contact_force_n``: env, step, part, world position,
  force magnitude.
- ``keep``: the trial ran its full window without an unexpected reset or a
  pre-fault fall.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from go2_him.him.env_cfg import LIN_VEL_SCALE
from go2_him.him.modules import HIMActor
from heat_bench.envs_mjlab.fault_impact import (
  FAULT_ELEVATION_SENSOR,
  TRUNK_GROUPS,
  body_group,
  body_part,
  fire_step,
  terrain_tile,
  tilt_angle,
)
from heat_bench.envs_mjlab.joint_fault_injection import (
  FAULT_CONTACT_SENSOR,
  contacts_in_base_frame,
)
from heat_bench.scripts.policy_loading import load_runner

from failbench.commands import KnotVelocityCommand
from failbench.env_cfg import DEFAULT_TASK, failbench_env_cfg
from failbench.faults import FAULT_PARAMS, FAULT_TERM, inject_kinodynamic_fault
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import GridPatternCfg, RayCastSensor
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class GenerateConfig:
  checkpoint_file: str
  """The frozen HIM policy (``policies/go2_him_rough_original_7200``)."""
  output_dir: str
  task: str = DEFAULT_TASK
  terrain: str = "rough"
  """A heat_bench ``FAULT_TERRAINS`` preset: rough, stairs, stairs_up, course."""
  terrain_seed: int | None = None
  """Terrain generator seed (also seeds torch); give each job its own."""
  num_envs: int = 4096
  rounds: int = 1
  device: str | None = None
  fault_time_s: tuple[float, float] = (3.0, 5.0)
  pre_fault_s: float = 1.5
  """Logged before the fault: past sensor poses for the fused map."""
  horizon_s: float = 7.0
  """Logged after the fault: the last snapshot plus the 5 s target window."""
  snapshot_s: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0)
  min_contact_force_n: float = 1.0
  prefix: str | None = None
  """Shard name prefix; defaults to '<terrain>_s<seed>'."""


def _np(x: torch.Tensor) -> np.ndarray:
  return x.detach().cpu().numpy()


def run_round(
  cfg: GenerateConfig, env: ManagerBasedRlEnv, wrapped, policy, actor
) -> dict[str, np.ndarray]:
  device = env.device
  n, step_dt = env.num_envs, env.step_dt
  robot = env.scene["robot"]
  fault = env.event_manager.get_term_cfg(FAULT_TERM).func
  assert isinstance(fault, inject_kinodynamic_fault)
  command = env.command_manager.get_term("twist")
  assert isinstance(command, KnotVelocityCommand)
  sensor = env.scene[FAULT_CONTACT_SENSOR]
  elev = env.scene[FAULT_ELEVATION_SENSOR]
  assert isinstance(elev, RayCastSensor)
  assert isinstance(elev.cfg.pattern, GridPatternCfg)
  assert env.scene.terrain is not None
  gen = env.scene.terrain.cfg.terrain_generator
  assert gen is not None
  num_cells = elev.data.distances.shape[-1]
  side = int(round(num_cells**0.5))

  geom_names = list(sensor.primary_names)
  part_names = list(dict.fromkeys(body_part(g) for g in geom_names))
  num_slots = sensor.cfg.num_slots
  slot_geom = torch.arange(len(geom_names) * num_slots, device=device) // num_slots
  geom_part = torch.tensor(
    [part_names.index(body_part(g)) for g in geom_names], device=device
  )
  slot_part = geom_part[slot_geom]
  groups = [body_group(p) for p in part_names]
  slot_trunk = torch.tensor([g in TRUNK_GROUPS for g in groups], device=device)[
    slot_part
  ]
  slot_logged = torch.tensor([g != "foot" for g in groups], device=device)[slot_part]

  pre = round(cfg.pre_fault_s / step_dt)
  hor = round(cfg.horizon_s / step_dt)
  num_t = pre + hor + 1
  snap_rel = torch.tensor([round(s / step_dt) for s in cfg.snapshot_s], device=device)
  num_j = len(fault.joint_names)

  def buf(*shape: int, dtype=torch.float32) -> torch.Tensor:
    return torch.zeros(n, num_t, *shape, dtype=dtype, device=device)

  dense = {
    "cmd": buf(3),
    "lin_vel_b": buf(3),
    "him_vel_est": buf(3),
    "ang_vel_b": buf(3),
    "projected_gravity": buf(3),
    "base_pos_w": buf(3),
    "base_quat_w": buf(4),
    "joint_pos": buf(num_j),
    "joint_vel": buf(num_j),
    "him_latent": buf(actor.latent_dim),
    "tilt": buf(),
    "trunk_contact": buf(dtype=torch.bool),
    "valid": buf(dtype=torch.bool),
  }
  maps = torch.full(
    (n, len(snap_rel), side, side), float("nan"), dtype=torch.float16, device=device
  )
  rows: dict[str, list[torch.Tensor]] = {
    k: [] for k in ("env_id", "step", "part", "pos_w", "force")
  }
  tile = torch.full((n, 2), -1, dtype=torch.long, device=device)
  pre_fault_fall = torch.zeros(n, dtype=torch.bool, device=device)
  reset_seen = torch.zeros(n, dtype=torch.bool, device=device)
  done = torch.zeros(n, dtype=torch.bool, device=device)

  obs, _ = wrapped.reset()
  prev_len = env.episode_length_buf.clone()
  fire = fire_step(fault.fault_time_s, step_dt)
  max_steps = int(fire.max().item()) + hor + 5
  for _ in range(max_steps):
    with torch.no_grad():
      obs, _, _, _ = wrapped.step(policy(obs))
      vel_est, latent = actor.estimate(obs)
    # Terminations are stripped, so a shrinking counter is an unexpected reset.
    reset_seen |= env.episode_length_buf < prev_len
    prev_len = env.episode_length_buf.clone()

    rel = env.episode_length_buf - fire
    j = rel + pre
    log = ~done & (j >= 0) & (j < num_t)
    if not log.any():
      if done.all():
        break
      continue
    e = log.nonzero().squeeze(-1)
    data = robot.data

    _, _, pos_w, force_w, valid = contacts_in_base_frame(sensor, robot)
    fmag = force_w.norm(dim=-1)
    valid = valid & (fmag >= cfg.min_contact_force_n)
    trunk_hit = (valid & slot_trunk).any(dim=-1)
    step_values = {
      "cmd": command.command,
      "lin_vel_b": data.root_link_lin_vel_b,
      "him_vel_est": vel_est / LIN_VEL_SCALE,
      "ang_vel_b": data.root_link_ang_vel_b,
      "projected_gravity": data.projected_gravity_b,
      "base_pos_w": data.root_link_pos_w,
      "base_quat_w": data.root_link_quat_w,
      "joint_pos": data.joint_pos,
      "joint_vel": data.joint_vel,
      "him_latent": latent,
      "tilt": tilt_angle(data.projected_gravity_b),
      "trunk_contact": trunk_hit,
    }
    for k, v in step_values.items():
      dense[k][e, j[e]] = v[e].to(dense[k].dtype)
    dense["valid"][e, j[e]] = True

    hit = valid & slot_logged[None, :] & log[:, None]
    if hit.any():
      he, hc = hit.nonzero(as_tuple=True)
      rows["env_id"].append(he)
      rows["step"].append(j[he])
      rows["part"].append(slot_part[hc])
      rows["pos_w"].append(pos_w[he, hc])
      rows["force"].append(fmag[he, hc])

    snap = log[:, None] & (rel[:, None] == snap_rel[None, :])
    if snap.any():
      se, sk = snap.nonzero(as_tuple=True)
      z = elev.data.hit_pos_w[se, :, 2] - data.root_link_pos_w[se, 2:3]
      z = torch.where(elev.data.distances[se] < 0, float("nan"), z)
      maps[se, sk] = z.view(-1, side, side).half()

    at_fault = log & (rel == 0)
    if at_fault.any():
      ef = at_fault.nonzero().squeeze(-1)
      r, c = terrain_tile(data.root_link_pos_w[ef], gen)
      tile[ef] = torch.stack([r, c], dim=-1)
    pre_fault_fall |= log & (rel < 0) & trunk_hit
    done |= log & (rel >= hor)
    if done.all():
      break

  mismatch = fault.faulted & (fault.fault_step != fire)
  keep = done & ~reset_seen & ~pre_fault_fall & ~mismatch & fault.faulted
  for name, bad in (
    ("reset mid-run", reset_seen),
    ("fell before the fault", pre_fault_fall),
    ("faulted off the predicted step", mismatch),
    ("didn't finish", ~done),
  ):
    if bad.any():
      print(f"[WARN] Dropping {int(bad.sum())} trials that {name}.")

  contacts = {
    k: torch.cat(v) if v else torch.empty(0, device=device) for k, v in rows.items()
  }
  out = {f"dense_{k}": _np(v) for k, v in dense.items()}
  out |= {f"contact_{k}": _np(v) for k, v in contacts.items()}
  out |= {
    "map": _np(maps),
    "keep": _np(keep),
    "fault_params": _np(fault.params),
    "fault_family": _np(fault.family),
    "fault_time_s": _np(fault.fault_time_s),
    "cmd_knots": _np(command.knots),
    "cmd_style": _np(command.style),
    "tile_fault": _np(tile),
    "pre_steps": np.array(pre),
    "step_dt": np.array(step_dt),
    "snapshot_steps": _np(snap_rel),
    "map_resolution": np.array(elev.cfg.pattern.resolution),
    "part_names": np.array(part_names),
    "joint_names": np.array(fault.joint_names),
    "fault_param_names": np.array(FAULT_PARAMS),
    "knot_s": np.array(command.knot_cfg.knot_s),
  }
  return out


def generate(cfg: GenerateConfig) -> list[Path]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.terrain_seed is not None:
    torch.manual_seed(cfg.terrain_seed)
  env_cfg = failbench_env_cfg(
    cfg.task, cfg.terrain, cfg.terrain_seed, fault_time_s=cfg.fault_time_s
  )
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped, runner = load_runner(
    env, device, checkpoint_file=cfg.checkpoint_file, task_id=cfg.task
  )
  policy = runner.get_inference_policy(device=device)
  actor = runner.alg.get_policy()
  if not isinstance(actor, HIMActor):
    raise ValueError(f"{cfg.checkpoint_file} isn't a HIMLoco policy.")

  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  seed = "none" if cfg.terrain_seed is None else str(cfg.terrain_seed)
  prefix = cfg.prefix or f"{cfg.terrain}_s{seed}"
  meta = {k: np.array(str(v)) for k, v in asdict(cfg).items()}
  paths = []
  for r in range(cfg.rounds):
    out = run_round(cfg, env, wrapped, policy, actor)
    out |= {f"config_{k}": v for k, v in meta.items()}
    out["terrain"] = np.array(cfg.terrain)
    out["terrain_seed"] = np.array(-1 if cfg.terrain_seed is None else cfg.terrain_seed)
    path = out_dir / f"{prefix}_r{r}.npz"
    np.savez_compressed(path, allow_pickle=False, **out)
    kept = int(out["keep"].sum())
    print(f"[INFO] Round {r}: {kept}/{cfg.num_envs} trials kept -> {path}")
    paths.append(path)
  env.close()
  return paths


def main():
  cfg = tyro.cli(GenerateConfig)
  if not Path(cfg.checkpoint_file).is_file():
    print(f"No checkpoint at {cfg.checkpoint_file}.")
    sys.exit(1)
  generate(cfg)


if __name__ == "__main__":
  main()
