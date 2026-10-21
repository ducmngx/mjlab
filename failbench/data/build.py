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
- ``upright[k]``: no fall yet (trunk/head contact or tilt over
  ``fall_tilt_deg``) at the end of the bin.
- contacts: non-foot contacts in the bin on the start map grid, as CSR rows
  (``contact_ptr`` per sample into ``contact_bin/ix/iy/part``); contacts
  off the map set ``contact_offmap[k]``.

Inputs: the clean map (trunk-relative heights, NaN = no terrain), what the
head LiDAR could see of it (``known_single`` from this pose,
``known_fused`` from it and the poses ``fuse_back_s`` before; see
``failbench/sensing.py``), ``state`` (33: HIM's velocity estimate, angular
velocity, projected gravity, joint positions and velocities), the true
velocity, the HIM latent, and the encoded fault (72).

``split``: 0 train, 1 val, 2 test_terrain (unseen terrain seeds), 3
test_fault (severe faults on ``held_out_joints``, never trained on).

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

from failbench.faults import FaultFamily, encode_fault
from failbench.sensing import LidarCfg, sensor_poses, visible_cells
from mjlab.utils.lab_api.math import wrap_to_pi

SPLITS = ("train", "val", "test_terrain", "test_fault")
TERRAINS = ("rough", "stairs", "stairs_up", "course")


@dataclass(frozen=True)
class BuildConfig:
  raw: tuple[str, ...]
  """Raw shard files (``generate.py`` output)."""
  output_dir: str
  horizon_s: float = 5.0
  target_hz: float = 10.0
  fall_tilt_deg: float = 60.0
  fuse_back_s: tuple[float, ...] = (0.5, 1.0, 1.5)
  held_out_joints: tuple[str, ...] = ("RL_calf_joint",)
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
  held = [list(joints).index(j) for j in cfg.held_out_joints]
  healthy = np.array([1, 1, 1, 1, 1, math.pi], dtype=np.float32)
  touched = ~np.isclose(params, healthy).all(-1)  # [N, J]
  ood = (family == FaultFamily.SEVERE) & touched[:, held].any(-1)
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

  def dense(name: str) -> torch.Tensor:
    return torch.as_tensor(raw[f"dense_{name}"][idx], device=device)

  pos, quat = dense("base_pos_w"), dense("base_quat_w")
  cmd, lin, ang = dense("cmd"), dense("lin_vel_b"), dense("ang_vel_b")
  tilt, trunk = dense("tilt"), dense("trunk_contact")
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
  fallen_t = trunk | (tilt > math.radians(cfg.fall_tilt_deg))
  t_ar = torch.arange(num_t, device=device)
  post = t_ar[None, :] >= pre
  fall_step = torch.where(fallen_t & post, t_ar, num_t).min(dim=1).values  # [n]

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
    put("tilt", tilt[:, ends])
    put("upright", ends[None, :] < fall_step[:, None])
    put("valid", valid_t[:, ends])
    put("fallen_at_start", fall_step <= s)
    put("offset_s", torch.full((n,), snap * step_dt, device=device))

    # Clean map and what the LiDAR saw of it, now and over the fused poses.
    height = maps[:, si]
    put("map_clean", height.half())
    pose_steps = [s] + [max(s - b, 0) for b in back]
    p_pos = to_yaw_frame(pos[:, pose_steps], o_pos, o_quat)  # [n,P,3]
    rel_quat = _yaw_rotate_quat(quat[:, pose_steps], -o_yaw)
    s_pos, s_axis = sensor_poses(p_pos, rel_quat, cfg.lidar)
    b_yaw = wrap_to_pi(heading(quat[:, pose_steps]) - o_yaw[:, None])
    single, fused = [], []
    for c0 in range(0, n, cfg.chunk):
      sl = slice(c0, c0 + cfg.chunk)
      sp, sa, bp, by = (x[sl] for x in (s_pos, s_axis, p_pos, b_yaw))
      single.append(
        visible_cells(
          height[sl], sp[:, :1], sa[:, :1], bp[:, :1], by[:, :1], res, cfg.lidar
        )
      )
      fused.append(visible_cells(height[sl], sp, sa, bp, by, res, cfg.lidar))
    put("known_single", torch.cat(single))
    put("known_fused", torch.cat(fused))

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
    put("cmd_style", raw["cmd_style"][idx].astype(np.uint8))
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
    "fuse_back_s": list(cfg.fuse_back_s),
  }
  (shard / "meta.json").write_text(json.dumps(meta, indent=2))
  return shard


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
