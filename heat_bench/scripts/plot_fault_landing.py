"""Plot a ``fault_landing.py`` contact log.

One figure per (fault joint, fault mode) group, plus a cross-mode summary:

- 3D scatter of post-fault contact points in the trunk frame at that
  instant, colored by contact force magnitude, with force arrows on the
  strongest contacts and the trunk box for scale.
- Top-down map, in the trunk frame frozen at fault time, of where non-foot
  body parts touched the ground, plus where the base ended up.
- Per-env peak contact force histogram.
- Summary: share of post-fault contacts per body group, per fault mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# Go1 trunk_collision box half-extents (m), from go1.xml, for scale only.
TRUNK_HALF = np.array([0.13, 0.09, 0.05])
GROUPS = ("trunk", "head", "hip", "thigh", "calf", "foot")


@dataclass(frozen=True)
class PlotConfig:
  input_file: str = "heat_bench_fault_landing.npz"
  """Contact log written by fault_landing.py."""
  output_dir: str = "heat_bench_fault_landing_plots"
  """Directory for the PNGs."""
  num_arrows: int = 60
  """Force arrows drawn on the strongest contacts in each 3D plot."""


def _trunk_box(ax) -> None:
  x, y, z = TRUNK_HALF
  corners = np.array(
    [[sx * x, sy * y, sz * z] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
  )
  for i in range(8):
    for j in range(i + 1, 8):
      if np.sum(corners[i] != corners[j]) == 1:
        ax.plot(*zip(corners[i], corners[j], strict=True), color="0.3", lw=0.8)


def _plot_group(d, env_sel: np.ndarray, title: str, path: Path, num_arrows: int):
  c_env = d["contact_env_id"]
  post = env_sel[c_env] & (d["contact_t"] >= 0)
  pos_b, force_b = d["contact_pos_b"][post], d["contact_force_b"][post]
  pos_f = d["contact_pos_f"][post]
  group = d["geom_groups"][d["contact_geom"][post]]
  fmag = np.linalg.norm(force_b, axis=-1)

  fig = plt.figure(figsize=(17, 5.5))
  fig.suptitle(f"{title}  ({env_sel.sum()} envs, {post.sum()} contact samples)")

  ax = fig.add_subplot(1, 3, 1, projection="3d")
  sc = ax.scatter(
    *pos_b.T,
    c=fmag,
    cmap="viridis",
    s=4,
    vmin=0,
    vmax=np.percentile(fmag, 99) if fmag.size else 1,
  )
  if fmag.size:
    top = np.argsort(fmag)[-num_arrows:]
    scale = 0.3 / max(fmag[top].max(), 1e-6)
    ax.quiver(*pos_b[top].T, *(force_b[top] * scale).T, color="crimson", lw=0.6)
  _trunk_box(ax)
  ax.set(
    xlabel="x fwd (m)",
    ylabel="y left (m)",
    zlabel="z up (m)",
    title="Contacts in trunk frame (arrows: force on robot)",
  )
  ax.set_box_aspect((1, 1, 0.7))
  fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.12, label="|F| (N)")

  ax = fig.add_subplot(1, 3, 2)
  nonfoot = group != "foot"
  if nonfoot.any():
    hb = ax.hexbin(
      pos_f[nonfoot, 0], pos_f[nonfoot, 1], gridsize=30, cmap="magma_r", mincnt=1
    )
    fig.colorbar(hb, ax=ax, label="non-foot contact samples")
  end = d["end_base_pos_f"][env_sel]
  ax.scatter(
    end[:, 0], end[:, 1], marker="x", color="tab:blue", s=18, label="base at window end"
  )
  ax.add_patch(
    Rectangle(
      (-TRUNK_HALF[0], -TRUNK_HALF[1]),
      2 * TRUNK_HALF[0],
      2 * TRUNK_HALF[1],
      fill=False,
      color="0.3",
    )
  )
  ax.set(
    xlabel="x fwd at fault (m)",
    ylabel="y left at fault (m)",
    aspect="equal",
    title="Landing map, trunk frame at fault time",
  )
  ax.legend(loc="upper left", fontsize=8)

  ax = fig.add_subplot(1, 3, 3)
  ax.hist(d["peak_force_n"][env_sel], bins=20, color="tab:gray")
  touched = d["trunk_touched"][env_sel].mean()
  ax.set(
    xlabel="peak post-fault contact force (N)",
    ylabel="envs",
    title=f"Peak impact force (trunk/head touched: {touched:.0%})",
  )

  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def _plot_summary(d, kept: np.ndarray, path: Path) -> None:
  mode_names = d["mode_names"]
  post = kept[d["contact_env_id"]] & (d["contact_t"] >= 0)
  group = d["geom_groups"][d["contact_geom"]]
  mode = d["fault_mode"][d["contact_env_id"]]
  fig, ax = plt.subplots(figsize=(8, 4))
  width = 0.8 / len(mode_names)
  for i, name in enumerate(mode_names):
    sel = post & (mode == i)
    if not sel.any():
      continue
    share = [(group[sel] == g).mean() for g in GROUPS]
    ax.bar(np.arange(len(GROUPS)) + i * width, share, width, label=name)
  ax.set_xticks(np.arange(len(GROUPS)) + width * (len(mode_names) - 1) / 2, GROUPS)
  ax.set(
    ylabel="share of post-fault contact samples",
    yscale="log",
    title="Which body parts touch the ground after the fault",
  )
  ax.legend()
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def plot_fault_landing(cfg: PlotConfig) -> list[Path]:
  d = dict(np.load(cfg.input_file))
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  kept = d["env_kept"]
  joint_names, mode_names = d["joint_names"], d["mode_names"]
  # Group by the env's fault joint set (sorted, so order doesn't split groups).
  joint_key = np.array(
    ["+".join(sorted(joint_names[j] for j in row)) for row in d["fault_joint"]]
  )
  paths = []
  for key in np.unique(joint_key[kept]):
    for m, mode in enumerate(mode_names):
      env_sel = kept & (joint_key == key) & (d["fault_mode"] == m)
      if not env_sel.any():
        continue
      path = out_dir / f"{key}_{mode}.png"
      _plot_group(d, env_sel, f"{key} -- {mode}", path, cfg.num_arrows)
      paths.append(path)
  summary = out_dir / "summary_body_groups.png"
  _plot_summary(d, kept, summary)
  paths.append(summary)
  print(f"[INFO] Wrote {len(paths)} figures to {out_dir}")
  return paths


def main():
  plot_fault_landing(tyro.cli(PlotConfig))


if __name__ == "__main__":
  main()
