"""Render built FailBench samples for checking by eye.

One PNG per sample: the clean map, the LiDAR-realistic map (unknown cells
grey), all post-start contacts coloured by time over
the clean map with the 5 s path, and the Δv / upright targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from failbench.data.dataset import FailBenchDataset, MapMode
from failbench.faults import TORQUE, FaultFamily, faulted_joint


@dataclass(frozen=True)
class InspectConfig:
  root: str
  output_dir: str
  split: str = "train"
  num: int = 20
  map_mode: MapMode = "fused"
  seed: int = 0
  fallen_first: bool = True
  """Pick samples that fall within the window first."""


def main():
  cfg = tyro.cli(InspectConfig)
  ds = FailBenchDataset(
    cfg.root, cfg.split, cfg.map_mode, drop_fallen_at_start=True, seed=cfg.seed
  )
  rng = np.random.default_rng(cfg.seed)
  order = rng.permutation(len(ds))
  if cfg.fallen_first:
    fell = np.array(
      [not bool(ds.shards[s]["upright"][r].all()) for s, r in ds.index[order]]
    )
    order = np.concatenate([order[fell], order[~fell]])
  out = Path(cfg.output_dir)
  out.mkdir(parents=True, exist_ok=True)
  meta = ds.metas[0]
  res, cells, bin_s = meta["map_resolution"], ds.cells, meta["bin_s"]
  half = (cells - 1) / 2 * res
  ext = (-half - res / 2, half + res / 2, -half - res / 2, half + res / 2)
  t = (np.arange(ds.num_bins) + 1) * bin_s

  for n, i in enumerate(order[: cfg.num]):
    item = ds[int(i)]
    s, row = (int(x) for x in ds.index[i])
    clean = ds.shards[s]["map_clean"][row].astype(np.float32)
    seen = item["map"].numpy()
    pose = item["pose"].numpy()
    contact = item["contact"].numpy()
    vmin, vmax = np.nanpercentile(clean, 2), np.nanpercentile(clean, 98)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    kw = dict(origin="lower", extent=ext, cmap="terrain", vmin=vmin, vmax=vmax)
    ax[0].imshow(clean, **kw)
    ax[0].set_title("clean map (z rel. trunk)")
    shown = np.where(seen[1] > 0, seen[0], np.nan)
    ax[1].set_facecolor("0.6")
    ax[1].imshow(shown, **kw)
    ax[1].set_title(f"{cfg.map_mode} LiDAR map, known {seen[1].mean():.0%}")
    ax[2].imshow(clean, **kw, alpha=0.5)
    b, iy, ix = np.nonzero(contact)
    sc = ax[2].scatter(
      (ix - (cells - 1) / 2) * res,
      (iy - (cells - 1) / 2) * res,
      c=(b + 1) * bin_s,
      s=8,
      cmap="plasma",
      vmin=0,
      vmax=t[-1],
    )
    ax[2].plot(pose[:, 0], pose[:, 1], "k-", lw=1.5)
    ax[2].plot([0], [0], "k^", ms=8)
    fig.colorbar(sc, ax=ax[2], label="contact time (s)")
    ax[2].set_title(
      f"non-foot contacts, offmap bins {int(item['contact_offmap'].sum())}"
    )
    for a in ax[:3]:
      a.set_xlim(ext[0], ext[1])
      a.set_ylim(ext[2], ext[3])
      a.set_aspect("equal")
    dv = item["dv"].numpy()
    for k, name in enumerate(("dvx", "dvy", "dwz")):
      ax[3].plot(t, dv[:, k], label=name)
    up = item["upright"].numpy()
    ax[3].plot(t, up.astype(float), "k--", label="upright")
    hit = item["collision"].numpy()
    ax[3].plot(t[hit], np.full(hit.sum(), 0.9), "rx", ms=4, label="collision")
    cmd = item["cmd_seq"].numpy()
    ax[3].step(t, cmd[:, 0], "C0:", where="pre", label="cmd vx")
    ax[3].legend(fontsize=8)
    ax[3].set_xlabel("time after start (s)")
    fam = FaultFamily(int(item["fault_family"])).name
    params = ds.shards[s]["fault_params"][row]
    joint = int(faulted_joint(torch.as_tensor(params)))
    meta_s = ds.metas[s]
    what = fam
    if joint >= 0:
      torque = params[joint, TORQUE]
      what += f" {meta_s['joint_names'][joint]}"
      what += "" if fam == "LOCK" else f" torque {torque:.2f}"
    fig.suptitle(
      f"{meta_s['terrain']} seed {meta_s['terrain_seed']},"
      f" +{float(item['offset_s']):.1f} s after fault, {what}"
    )
    fig.tight_layout()
    path = out / f"sample_{n:03d}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
  print(f"[INFO] Wrote {min(cfg.num, len(order))} renders to {out}")


if __name__ == "__main__":
  main()
