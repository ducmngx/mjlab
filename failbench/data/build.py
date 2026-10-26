"""Raw FailBench shards -> training sample shards.

Every kept trial gives one sample per map snapshot (``snapshot_s`` in
``generate.py``): the inputs at that instant and the next ``horizon_s``
(5 s) of targets at ``target_hz`` (10 Hz). For a sample starting at
control step ``s`` with ``r`` control steps per bin, bin ``k`` covers steps
``s + r k + 1 .. s + r (k + 1)``:

- ``cmd_seq[k]``: the command the policy tracked in that bin (``cmd`` at
  ``s + r k``; knots switch on bin edges).
- ``dv[k]``: mean over the bin of (vx, vy, wz) minus the command the step
  was taken under, in the body frame of the moment.
- ``pose[k]``: (x, y, yaw) at the end of the bin in the start yaw frame.
- ``cmd_path[k]``: the same for the commanded twist integrated from the
  start pose, so ``pose - cmd_path`` is the drift from the plan.
- ``upright[k]``: no fall yet (``fallen``: tilt over ``fall_tilt_deg`` or
  the trunk under ``fall_height_m`` above the ground) at the end of the bin.
- ``safe[k]``: not yet past the point of no return (``no_return_step``):
  the last moment the robot was still as upright as it walked before the
  fault, after which it went down for good. Earlier than ``upright``'s fall
  by however long the fall takes, so a fall predictor trained on it warns
  while there is still time to act.
- ``collision[k]``: the trunk or head touched terrain or an obstacle in the
  bin (walls on the course, a step edge) whether or not the robot fell.
- contacts: non-foot contacts in the bin on the start map grid, as CSR rows
  (``contact_ptr`` per sample into ``contact_bin/ix/iy/part``); contacts
  off the map set ``contact_offmap[k]``.

Inputs: the clean map (trunk-relative heights, NaN = no terrain), what the
head LiDAR could see of it (``known_single`` from this pose,
``known_fused`` from it and the poses ``fuse_back_s`` before; see
``failbench/sensing.py``), ``state`` (33: HIM's velocity estimate, angular
velocity, projected gravity, joint positions and velocities), the true
velocity, the HIM latent, ``history`` (the last ``history_s`` at
``target_hz``, oldest first, ending at the start: ``state`` plus the
command, 36 per step), and the encoded fault (24). ``plan_hold``: the
post-fault plan holds the fault-time command.

``split``: 0 train, 1 val, 2 test_terrain (unseen terrain seeds), 3
test_fault (the ``held_out`` joint and family pairs, never trained on).

Each shard is a directory of ``.npy`` files (memory-mappable) plus
``meta.json``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
from heat_bench.envs_mjlab.fault_impact import heading, to_yaw_frame

from failbench.faults import FaultFamily, encode_fault, faulted_joint
from failbench.sensing import LidarCfg, sensor_poses, visible_cells
from mjlab.utils.lab_api.math import wrap_to_pi

SPLITS = ("train", "val", "test_terrain", "test_fault")
TERRAINS = ("rough", "stairs", "stairs_up", "course", "stairs_tall", "stairs_up_tall")
FALL_TILT_DEG = 60.0
FALL_HEIGHT_M = 0.15
"""Trunk height above the ground (nominal 0.30) below which it is down."""


def no_return_step(
  tilt: torch.Tensor,
  base_height: torch.Tensor,
  fall_step: torch.Tensor,
  fault_step: int,
  tilt_margin_deg: float,
  height_margin_m: float,
  ref_steps: int,
) -> torch.Tensor:
  """``[n]`` first step from which the robot never again walks as upright
  as it did before the fault (tilt within ``tilt_margin_deg`` of its worst
  over the ``ref_steps`` before it, trunk within ``height_margin_m`` of its
  lowest), until its fall at ``fall_step`` (``[n]``, ``T`` for none: then
  ``T`` too). Relative to the robot's own pre-fault walk, so a trunk pitched
  on a stair flight isn't counted. At the fault step if it never was."""
  t = torch.arange(tilt.shape[1], device=tilt.device)
  ref = slice(fault_step - ref_steps, fault_step)
  tilt_ok = tilt[:, ref].amax(1, keepdim=True) + math.radians(tilt_margin_deg)
  height_ok = base_height[:, ref].amin(1, keepdim=True) - height_margin_m
  ok = (tilt <= tilt_ok) & (base_height >= height_ok)
  ok &= (t >= fault_step) & (t < fall_step[:, None])
  last_ok = torch.where(ok, t, fault_step - 1).amax(1)
  return torch.where(fall_step < tilt.shape[1], last_ok + 1, fall_step)


def fallen(
  tilt: torch.Tensor,
  base_height: torch.Tensor,
  tilt_deg: float = FALL_TILT_DEG,
  height_m: float = FALL_HEIGHT_M,
) -> torch.Tensor:
  """The robot is down: tipped over, or its trunk on (or near) the ground.
  Trunk contact alone isn't a fall: on the course it is often a wall."""
  return (tilt > math.radians(tilt_deg)) | (base_height < height_m)


@dataclass(frozen=True)
class BuildConfig:
  raw: tuple[str, ...]
  """Raw shard files (``generate.py`` output)."""
  output_dir: str
  horizon_s: float = 5.0
  target_hz: float = 10.0
  fall_tilt_deg: float = FALL_TILT_DEG
  fall_height_m: float = FALL_HEIGHT_M
  no_return_tilt_margin_deg: float = 15.0
  no_return_height_margin_m: float = 0.06
  """``safe``: how far past its pre-fault walk the robot may tip or sink and
  still count as upright (``no_return_step``)."""
  fuse_back_s: tuple[float, ...] = (0.5, 1.0, 1.5)
  history_s: float = 2.0
  held_out: tuple[str, ...] = ("RL_calf_joint:LOCK",)
  """``joint:FAMILY`` pairs held out as test_fault."""
  val_seed_mod: int = 1
  test_seed_mod: int = 0
  """Terrain seeds with ``seed % 10`` equal to these go to val / test_terrain."""
  chunk: int = 16
  """Samples per visibility batch (GPU memory)."""
  device: str | None = None
  lidar: LidarCfg = field(default_factory=LidarCfg)


def _split(
  seed: int, family: np.ndarray, params: np.ndarray, joints, cfg
) -> np.ndarray:
  base = 0
  if seed >= 0 and seed % 10 == cfg.test_seed_mod:
    base = 2
  elif seed >= 0 and seed % 10 == cfg.val_seed_mod:
    base = 1
  split = np.full(len(family), base, dtype=np.uint8)
  joint = faulted_joint(torch.as_tensor(params)).numpy()
  for pair in cfg.held_out:
    name, fam = pair.split(":")
    ood = (family == FaultFamily[fam]) & (joint == list(joints).index(name))
    split[ood] = 3
  return split


def build_shard(path: Path, cfg: BuildConfig, device: str) -> Path:
  raw = np.load(path)
  keep = raw["keep"]
  idx = np.flatnonzero(keep)
  pre = int(raw["pre_steps"])
  step_dt = float(raw["step_dt"])
  r = round(1.0 / (cfg.target_hz * step_dt))
  nb = round(cfg.horizon_s * cfg.target_hz)
  snaps = raw["snapshot_steps"]
  res = float(raw["map_resolution"])
  cells = raw["map"].shape[-1]
  back = [round(b / step_dt) for b in cfg.fuse_back_s]
  num_hist = round(cfg.history_s * cfg.target_hz)
  if (num_hist - 1) * r > pre:
    raise ValueError(
      f"{path} logs {pre * step_dt} s before the fault, less than "
      f"the {cfg.history_s} s history."
    )

  def dense(name: str) -> torch.Tensor:
    return torch.as_tensor(raw[f"dense_{name}"][idx], device=device)

  pos, quat = dense("base_pos_w"), dense("base_quat_w")
  cmd, lin, ang = dense("cmd"), dense("lin_vel_b"), dense("ang_vel_b")
  tilt, trunk = dense("tilt"), dense("trunk_contact")
  base_height = dense("base_height")
  state_all = torch.cat(
    [
      dense("him_vel_est"),
      ang,
      dense("projected_gravity"),
      dense("joint_pos"),
      dense("joint_vel"),
    ],
    dim=-1,
  )
  latent = dense("him_latent")
  valid_t = dense("valid")
  n, num_t = valid_t.shape
  fallen_t = fallen(tilt, base_height, cfg.fall_tilt_deg, cfg.fall_height_m)
  t_ar = torch.arange(num_t, device=device)
  post = t_ar[None, :] >= pre
  fall_step = torch.where(fallen_t & post, t_ar, num_t).min(dim=1).values  # [n]
  nr_step = no_return_step(
    tilt,
    base_height,
    fall_step,
    pre,
    cfg.no_return_tilt_margin_deg,
    cfg.no_return_height_margin_m,
    round(1.0 / step_dt),
  )

  maps = torch.as_tensor(raw["map"][idx].astype(np.float32), device=device)
  params = raw["fault_params"][idx]
  family = raw["fault_family"][idx]
  seed = int(raw["terrain_seed"])
  split = _split(seed, family, params, raw["joint_names"], cfg)
  fault_enc = encode_fault(torch.as_tensor(params)).numpy().astype(np.float32)

  # Contact rows of kept trials, re-indexed to 0..n-1.
  remap = np.full(len(keep), -1)
  remap[idx] = np.arange(n)
  c_env = remap[raw["contact_env_id"].astype(int)]
  c_ok = c_env >= 0
  c_env = torch.as_tensor(c_env[c_ok], device=device)
  c_step = torch.as_tensor(raw["contact_step"][c_ok], device=device)
  c_part = torch.as_tensor(raw["contact_part"][c_ok], device=device)
  c_pos = torch.as_tensor(raw["contact_pos_w"][c_ok], device=device)

  bins = torch.arange(nb, device=device)
  out: dict[str, list[np.ndarray]] = {}

  def put(name: str, value: torch.Tensor | np.ndarray) -> None:
    if isinstance(value, torch.Tensor):
      value = value.detach().cpu().numpy()
    out.setdefault(name, []).append(value)

  contact_rows: list[tuple[np.ndarray, ...]] = []
  for si, snap in enumerate(snaps.tolist()):
    s = pre + snap
    if s + r * nb >= num_t:
      raise ValueError(f"Snapshot at step {snap} has no {cfg.horizon_s} s window.")
    ends = s + r * (bins + 1)  # [nb]
    o_pos, o_quat = pos[:, s], quat[:, s]
    o_yaw = heading(o_quat)

    put("state", state_all[:, s])
    put("state_true_vel", lin[:, s])
    put("him_latent", latent[:, s])
    put("cmd_seq", cmd[:, s + r * bins])
    steps = s + 1 + r * bins[:, None] + torch.arange(r, device=device)[None, :]
    v = torch.cat([lin[:, steps, :2], ang[:, steps, 2:]], dim=-1)  # [n,nb,r,3]
    put("dv", (v - cmd[:, steps - 1]).mean(dim=2))
    p = to_yaw_frame(pos[:, ends], o_pos, o_quat)
    yaw = wrap_to_pi(heading(quat[:, ends]) - o_yaw[:, None])
    put("pose", torch.cat([p[..., :2], yaw[..., None]], dim=-1))
    put("cmd_path", _integrate(cmd[:, s : s + r * nb], step_dt)[:, r - 1 :: r])
    hist = s - r * torch.arange(num_hist - 1, -1, -1, device=device)
    put("history", torch.cat([state_all[:, hist], cmd[:, hist]], dim=-1))
    put("tilt", tilt[:, ends])
    put("base_height", base_height[:, ends])
    put("collision", trunk[:, steps].any(dim=-1))
    put("upright", ends[None, :] < fall_step[:, None])
    put("safe", ends[None, :] < nr_step[:, None])
    put("valid", valid_t[:, ends])
    put("fallen_at_start", fall_step <= s)
    put("offset_s", torch.full((n,), snap * step_dt, device=device))

    # Clean map and what the LiDAR saw of it, now and over the fused poses.
    height = maps[:, si]
    put("map_clean", height.half())
    pose_steps = [s] + [max(s - b, 0) for b in back]
    single, fused = lidar_known(
      height, pos[:, pose_steps], quat[:, pose_steps], res, cfg.lidar, cfg.chunk
    )
    put("known_single", single)
    put("known_fused", fused)

    # Contacts in bins (s, s + r * nb] on this map.
    in_win = (c_step > s) & (c_step <= s + r * nb)
    ce = c_env[in_win]
    cb = (c_step[in_win] - s - 1) // r
    cp = to_yaw_frame(c_pos[in_win][:, None], o_pos[ce], o_quat[ce])[:, 0]
    ix = torch.round(cp[:, 0] / res + (cells - 1) / 2).long()
    iy = torch.round(cp[:, 1] / res + (cells - 1) / 2).long()
    on = (ix >= 0) & (ix < cells) & (iy >= 0) & (iy < cells)
    offmap = torch.zeros(n, nb, dtype=torch.bool, device=device)
    offmap[ce[~on], cb[~on]] = True
    put("contact_offmap", offmap)
    contact_rows.append(
      tuple(
        x.cpu().numpy() for x in (ce[on], cb[on], ix[on], iy[on], c_part[in_win][on])
      )
    )

    put("fault", fault_enc)
    put("fault_params", params.astype(np.float32))
    put("fault_family", family.astype(np.uint8))
    put("split", split)
    put("trial", idx.astype(np.int32))
    put("plan_hold", raw["plan_hold"][idx])
    put("tile", raw["tile_fault"][idx].astype(np.int16))

  # CSR contacts over samples ordered (snapshot, trial).
  ptr, cols = [0], {k: [] for k in ("bin", "ix", "iy", "part")}
  for ce, cb, ix, iy, cp in contact_rows:
    order = np.lexsort((cb, ce))
    counts = np.bincount(ce, minlength=n)
    ptr.extend((ptr[-1] + np.cumsum(counts)).tolist())
    for k, x in zip(("bin", "ix", "iy", "part"), (cb, ix, iy, cp), strict=True):
      cols[k].append(x[order].astype(np.uint8))

  shard = Path(cfg.output_dir) / path.stem
  shard.mkdir(parents=True, exist_ok=True)
  for k, v in out.items():
    np.save(shard / f"{k}.npy", np.concatenate(v))
  np.save(shard / "contact_ptr.npy", np.asarray(ptr, dtype=np.int64))
  for k, v in cols.items():
    np.save(
      shard / f"contact_{k}.npy", np.concatenate(v) if v else np.zeros(0, np.uint8)
    )
  terrain = str(raw["terrain"])
  meta = {
    "source": str(path),
    "num_samples": int(n * len(snaps)),
    "terrain": terrain,
    "terrain_id": TERRAINS.index(terrain) if terrain in TERRAINS else -1,
    "terrain_seed": seed,
    "num_bins": nb,
    "bin_s": r * step_dt,
    "map_cells": cells,
    "map_resolution": res,
    "snapshot_s": [x * step_dt for x in snaps.tolist()],
    "part_names": raw["part_names"].tolist(),
    "joint_names": raw["joint_names"].tolist(),
    "fault_param_names": raw["fault_param_names"].tolist(),
    "splits": list(SPLITS),
    "fall_tilt_deg": cfg.fall_tilt_deg,
    "fall_height_m": cfg.fall_height_m,
    "no_return_tilt_margin_deg": cfg.no_return_tilt_margin_deg,
    "no_return_height_margin_m": cfg.no_return_height_margin_m,
    "fuse_back_s": list(cfg.fuse_back_s),
  }
  (shard / "meta.json").write_text(json.dumps(meta, indent=2))
  return shard


def lidar_known(
  height: torch.Tensor,
  pos: torch.Tensor,
  quat: torch.Tensor,
  resolution: float,
  lidar: LidarCfg,
  chunk: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
  """What the head LiDAR saw of trunk-centred, yaw-aligned maps ``height [n,
  H, W]``: from the current pose only and fused over all poses. ``pos [n, P,
  3]`` / ``quat [n, P, 4]`` are base poses in any common frame, the current
  one (the map's) first."""
  o_pos, o_quat = pos[:, 0], quat[:, 0]
  o_yaw = heading(o_quat)
  p_pos = to_yaw_frame(pos, o_pos, o_quat)  # [n,P,3]
  rel_quat = _yaw_rotate_quat(quat, -o_yaw)
  s_pos, s_axis = sensor_poses(p_pos, rel_quat, lidar)
  b_yaw = wrap_to_pi(heading(quat) - o_yaw[:, None])
  single, fused = [], []
  for c0 in range(0, height.shape[0], chunk):
    sl = slice(c0, c0 + chunk)
    sp, sa, bp, by = (x[sl] for x in (s_pos, s_axis, p_pos, b_yaw))
    single.append(
      visible_cells(
        height[sl], sp[:, :1], sa[:, :1], bp[:, :1], by[:, :1], resolution, lidar
      )
    )
    fused.append(visible_cells(height[sl], sp, sa, bp, by, resolution, lidar))
  return torch.cat(single), torch.cat(fused)


def _integrate(cmd: torch.Tensor, dt: float) -> torch.Tensor:
  """(x, y, yaw) after each step ``[n, T, 3]`` of body-frame twists ``cmd
  [n, T, 3]`` held ``dt`` each, from the origin facing +x."""
  yaw = torch.cumsum(cmd[..., 2] * dt, dim=1)
  yaw0 = yaw - cmd[..., 2] * dt
  c, s = torch.cos(yaw0), torch.sin(yaw0)
  dx = (c * cmd[..., 0] - s * cmd[..., 1]) * dt
  dy = (s * cmd[..., 0] + c * cmd[..., 1]) * dt
  return torch.stack(
    [torch.cumsum(dx, 1), torch.cumsum(dy, 1), wrap_to_pi(yaw)], dim=-1
  )


def _yaw_rotate_quat(quat: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """Pre-multiply ``(w, x, y, z)`` quaternions ``[n, P, 4]`` by a yaw
  rotation ``[n]``: the orientation expressed in a frame yawed by -yaw."""
  half = yaw / 2
  qz = torch.stack(
    [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
    dim=-1,
  )[:, None, :].expand_as(quat)
  w1, x1, y1, z1 = qz.unbind(-1)
  w2, x2, y2, z2 = quat.unbind(-1)
  return torch.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    dim=-1,
  )


def main():
  cfg = tyro.cli(BuildConfig)
  device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
  for p in cfg.raw:
    with torch.no_grad():
      shard = build_shard(Path(p), cfg, device)
    print(f"[INFO] {p} -> {shard}")


if __name__ == "__main__":
  main()
