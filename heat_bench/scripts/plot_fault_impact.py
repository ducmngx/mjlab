"""Plot a ``fault_impact.py`` dataset.

Cross-group summaries:

- survival: share of trials still up vs time since the fault, per mode and
  (faulted trials only) per terrain type;
- velocity error ``dv_b`` mean +- std vs time since the fault, per mode,
  over steps before each trial's fall;
- first contact time after the fault, per body group, per mode.

One figure per group (fault mode, or fault joint x mode):

- path fan in the fault yaw frame: actual paths (red = fell, blue =
  survived) against the commanded ones (dashed);
- contact event raster: one row per trial, each event at its onset time
  after the fault, coloured by body group, sized by peak force;
- contact risk maps (share of trials touching each cell) over the mean
  elevation map, in the fault-time frame and in the fall-onset frame.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tyro  # noqa: E402

GROUP_COLORS = {
  "trunk": "tab:red",
  "head": "tab:purple",
  "hip": "tab:orange",
  "thigh": "tab:green",
  "calf": "tab:blue",
  "foot": "tab:gray",
}


@dataclass(frozen=True)
class PlotConfig:
  input_file: str = "heat_bench_fault_impact.npz"
  """Dataset written by fault_impact.py."""
  output_dir: str = "heat_bench_fault_impact_plots"
  """Directory for the PNGs."""
  group_by: Literal["mode", "joint_mode"] = "mode"
  """Per-group figures by fault mode, or by fault joint set and mode."""
  max_paths: int = 80
  """Paths drawn per path fan."""


def _nanmean(x: np.ndarray, axis: int) -> np.ndarray:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    return np.nanmean(x, axis=axis)


def _nanstd(x: np.ndarray, axis: int) -> np.ndarray:
  with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    return np.nanstd(x, axis=axis)


def _time_axis(d) -> np.ndarray:
  num_t = d["dense_valid"].shape[1]
  pre = round(float(d["pre_fault_s"]) / float(d["step_dt"]))
  return (np.arange(num_t) - pre) * float(d["step_dt"])


def _grid_extent(d) -> list[float]:
  (sx, sy), r = d["grid_size"], float(d["grid_resolution"])
  return [-sx / 2 - r / 2, sx / 2 + r / 2, -sy / 2 - r / 2, sy / 2 + r / 2]


def _upright_mask(d) -> np.ndarray:
  """Logged steps before each trial's fall, [B, T]."""
  ttf = np.where(d["fell"], d["time_to_fall_s"], np.inf)
  return d["dense_valid"] & (_time_axis(d)[None, :] < ttf[:, None])


def _plot_survival(d, kept: np.ndarray, path: Path) -> None:
  t = np.linspace(0.0, float(d["horizon_s"]), 300)
  fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
  ax = axes[0]
  for m, name in enumerate(d["mode_names"]):
    sel = kept & (d["fault_mode"] == m)
    if sel.any():
      ttf = np.where(d["fell"][sel], d["time_to_fall_s"][sel], np.inf)
      up = (ttf[None, :] > t[:, None]).mean(axis=1)
      ax.plot(t, up, label=f"{name} (n={sel.sum()})")
  ax.set(
    xlabel="time since fault (s)",
    ylabel="share still up",
    ylim=(0, 1.02),
    title="Survival by fault mode",
  )
  ax.legend()

  ax = axes[1]
  faulted = kept & (d["mode_names"][d["fault_mode"]] != "none")
  col = d["terrain_tile_fault"][:, 1]
  for c, name in enumerate(d["terrain_names"]):
    sel = faulted & (col == c)
    if sel.any():
      ttf = np.where(d["fell"][sel], d["time_to_fall_s"][sel], np.inf)
      ax.plot(
        t, (ttf[None, :] > t[:, None]).mean(axis=1), label=f"{name} (n={sel.sum()})"
      )
  ax.set(
    xlabel="time since fault (s)",
    ylabel="share still up",
    ylim=(0, 1.02),
    title="Survival by terrain at fault (faulted trials)",
  )
  ax.legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def _plot_dv(d, kept: np.ndarray, path: Path) -> None:
  t = _time_axis(d)
  dv = np.where(_upright_mask(d)[..., None], d["dense_dv_b"], np.nan)
  labels = ("vx - cmd (m/s)", "vy - cmd (m/s)", "wz - cmd (rad/s)")
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), sharex=True)
  for m, name in enumerate(d["mode_names"]):
    sel = kept & (d["fault_mode"] == m)
    if not sel.any():
      continue
    mean, std = _nanmean(dv[sel], 0), _nanstd(dv[sel], 0)
    for k, ax in enumerate(axes):
      (line,) = ax.plot(t, mean[:, k], label=name, lw=1.2)
      ax.fill_between(
        t,
        mean[:, k] - std[:, k],
        mean[:, k] + std[:, k],
        color=line.get_color(),
        alpha=0.15,
      )
  for k, ax in enumerate(axes):
    ax.axvline(0.0, color="0.3", lw=0.8, ls=":")
    ax.set(xlabel="time since fault (s)", ylabel=labels[k])
  axes[0].legend()
  fig.suptitle("Body velocity error vs command (mean +- std, upright steps only)")
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def _plot_first_contact(d, kept: np.ndarray, path: Path) -> None:
  groups = [g for g in d["group_names"] if g != "foot"]
  fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.8), sharey=True)
  bins = np.linspace(0.0, float(d["horizon_s"]), 31)
  for ax, g in zip(axes, groups, strict=True):
    gi = list(d["group_names"]).index(g)
    for m, name in enumerate(d["mode_names"]):
      sel = kept & (d["fault_mode"] == m)
      x = d["first_contact_dt_group"][sel, gi]
      x = x[np.isfinite(x)]
      if x.size:
        ax.hist(x, bins=bins, histtype="step", lw=1.4, label=f"{name} ({x.size})")
    ax.set(xlabel="first contact after fault (s)", title=g)
    ax.legend(fontsize=7)
  axes[0].set_ylabel("trials")
  fig.suptitle("When each body group first touches the terrain")
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def _risk_panel(fig, ax, d, sel: np.ndarray, frame: str, title: str) -> None:
  risk = d[f"risk_{frame}"][sel]
  elev = _nanmean(d[f"elevation_{frame}"][sel], 0)
  extent = _grid_extent(d)
  share = (risk[:, 0] > 0).mean(axis=0) if risk.size else np.zeros(elev.shape)
  if np.isfinite(elev).any():
    ax.contour(
      elev, levels=12, colors="0.55", linewidths=0.5, origin="lower", extent=extent
    )
  im = ax.imshow(
    np.ma.masked_equal(share, 0.0),
    origin="lower",
    extent=extent,
    cmap="magma_r",
    vmin=0,
    vmax=max(float(np.max(share)), 1e-6),
  )
  fig.colorbar(im, ax=ax, shrink=0.8, label="share of trials touching cell")
  outside = d[f"risk_outside_{frame}"][sel].sum()
  ax.set(
    xlabel="x fwd (m)",
    ylabel="y left (m)",
    title=f"{title}\n(contour: mean terrain; {outside} contacts off-map)",
  )


def _plot_group(d, sel: np.ndarray, title: str, path: Path, max_paths: int) -> None:
  t = _time_axis(d)
  fig, axes = plt.subplots(1, 4, figsize=(24, 5.8))
  fig.suptitle(f"{title}  ({sel.sum()} trials, fell {d['fell'][sel].mean():.0%})")

  ax = axes[0]
  envs = np.flatnonzero(sel)[:max_paths]
  post = t >= 0
  for b in envs:
    color = "tab:red" if d["fell"][b] else "tab:blue"
    act, exp = d["dense_actual_fy"][b, post], d["dense_expected_fy"][b, post]
    ok = d["dense_valid"][b, post]
    ax.plot(act[ok, 0], act[ok, 1], color=color, lw=0.7, alpha=0.6)
    ax.plot(exp[ok, 0], exp[ok, 1], color="0.5", lw=0.5, ls="--", alpha=0.5)
  ax.plot([], [], color="tab:red", label="actual (fell)")
  ax.plot([], [], color="tab:blue", label="actual (survived)")
  ax.plot([], [], color="0.5", ls="--", label="commanded")
  ax.set(
    xlabel="x fwd at fault (m)",
    ylabel="y left at fault (m)",
    aspect="equal",
    title="Paths after the fault, fault yaw frame",
  )
  ax.legend(fontsize=8)

  ax = axes[1]
  ev_sel = sel[d["event_env_id"]] & (d["event_onset_dt"] >= 0)
  ev_env = d["event_env_id"][ev_sel]
  # Rows ordered by each trial's first event, trials without events last.
  first = np.full(len(sel), np.inf)
  np.minimum.at(first, ev_env, d["event_onset_dt"][ev_sel])
  order = np.argsort(first[sel], kind="stable")
  row_of = np.empty(len(sel), dtype=int)
  row_of[np.flatnonzero(sel)[order]] = np.arange(sel.sum())
  groups = d["group_names"][d["event_group"][ev_sel]]
  force = d["event_peak_force_n"][ev_sel]
  size = 4 + 40 * force / max(force.max(), 1.0) if force.size else force
  for g, color in GROUP_COLORS.items():
    m = groups == g
    if m.any():
      ax.scatter(
        d["event_onset_dt"][ev_sel][m],
        row_of[ev_env[m]],
        s=size[m],
        color=color,
        label=g,
      )
  ttf = d["time_to_fall_s"][sel][order]
  ax.scatter(ttf, np.arange(sel.sum()), marker="|", color="k", s=30, label="fall onset")
  ax.set(
    xlabel="time since fault (s)",
    ylabel="trial (sorted by first contact)",
    title="Contact events: which part, when (size = peak force)",
  )
  ax.legend(fontsize=7, loc="lower right")

  _risk_panel(fig, axes[2], d, sel, "fy", "Contact risk, fault-time frame")
  _risk_panel(fig, axes[3], d, sel & d["fell"], "fl", "Contact risk, fall-onset frame")
  fig.tight_layout()
  fig.savefig(path, dpi=130)
  plt.close(fig)


def plot_fault_impact(cfg: PlotConfig) -> list[Path]:
  d = dict(np.load(cfg.input_file))
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  kept = d["env_kept"]
  mode_names, joint_names = d["mode_names"], d["joint_names"]
  paths = []
  for name, fn in (
    ("summary_survival.png", _plot_survival),
    ("summary_velocity_error.png", _plot_dv),
    ("summary_first_contact.png", _plot_first_contact),
  ):
    fn(d, kept, out_dir / name)
    paths.append(out_dir / name)

  if cfg.group_by == "mode":
    key = mode_names[d["fault_mode"]]
  else:
    joints = ["+".join(sorted(joint_names[j] for j in row)) for row in d["fault_joint"]]
    key = np.array(
      [f"{j}_{m}" for j, m in zip(joints, mode_names[d["fault_mode"]], strict=True)]
    )
  for k in np.unique(key[kept]):
    sel = kept & (key == k)
    path = out_dir / f"{k}.png"
    _plot_group(d, sel, str(k), path, cfg.max_paths)
    paths.append(path)
  print(f"[INFO] Wrote {len(paths)} figures to {out_dir}")
  return paths


def main():
  plot_fault_impact(tyro.cli(PlotConfig))


if __name__ == "__main__":
  main()
