"""Render fault trials with their recorded data drawn into the scene.

Called by ``fault_impact.py`` at the end of a run (``--render-trials``),
while the env and its terrain still exist: the robot pose snapshotted at
the fault and at each window end is re-posed on a CPU copy of the model and
rendered offscreen, with the trial's data added as 3D geoms. One camera per
trial, fixed at fault time, so the frames line up:

- elevation map: translucent tiles on the 4x4 m grid around the fault
  pose, coloured by height (blue low -> red high);
- trunk impact cells so far: raised boxes coloured by the second after the
  fault they were first touched in (yellow = 0-1 s ... purple = 4-5 s);
- trunk path so far (blue, red once fallen) and the commanded path (grey);
- black arrow: trunk position and heading at the fault; the faulted joint
  as a sphere (red = lock, blue = free).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

WINDOW_COLORS = np.array(
  [
    [1.0, 1.0, 0.0],  # 0-1 s yellow
    [1.0, 0.5, 0.0],  # 1-2 s orange
    [0.9, 0.0, 0.0],  # 2-3 s red
    [0.9, 0.0, 0.9],  # 3-4 s magenta
    [0.3, 0.0, 0.9],  # 4-5 s purple
  ]
)
MODE_COLORS = {"lock": (1.0, 0.1, 0.1, 1.0), "free": (0.2, 0.5, 1.0, 1.0)}
_HEIGHT_RAMP = np.array(
  [[0.1, 0.2, 0.9], [0.0, 0.8, 0.9], [0.1, 0.8, 0.2], [1.0, 0.9, 0.1], [0.9, 0.1, 0.1]]
)


def _yaw(quat: np.ndarray) -> float:
  w, x, y, z = quat
  return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rot_z(yaw: float) -> np.ndarray:
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _add(
  scene: mujoco.MjvScene, kind, size, pos, mat, rgba, emission: float = 0.0
) -> None:
  if scene.ngeom >= scene.maxgeom:
    return
  mujoco.mjv_initGeom(
    scene.geoms[scene.ngeom],
    kind,
    np.asarray(size, dtype=np.float64),
    np.asarray(pos, dtype=np.float64),
    np.asarray(mat, dtype=np.float64).flatten(),
    np.asarray(rgba, dtype=np.float32),
  )
  scene.geoms[scene.ngeom].emission = emission
  scene.ngeom += 1


def _height_rgb(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
  u = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * (len(_HEIGHT_RAMP) - 1)
  i = np.floor(u).astype(int).clip(max=len(_HEIGHT_RAMP) - 2)
  return _HEIGHT_RAMP[i] + (u - i)[:, None] * (_HEIGHT_RAMP[i + 1] - _HEIGHT_RAMP[i])


def render_trial(
  model: mujoco.MjModel,
  renderer: mujoco.Renderer,
  out: dict[str, np.ndarray],
  b: int,
  jnt_ids: dict[str, int],
) -> list[np.ndarray]:
  """Frames for trial ``b``: at the fault and at each window end."""
  data = mujoco.MjData(model)
  f_pos = out["fault_base_pos_w"][b]
  yaw = _yaw(out["fault_base_quat_w"][b])
  rot = _rot_z(yaw)
  mode = str(out["mode_names"][out["fault_mode"][b]])
  joints = [str(out["joint_names"][j]) for j in out["fault_joint"][b]]
  step_dt = float(out["step_dt"])
  pre = round(float(out["pre_fault_s"]) / step_dt)
  edges = out["window_edges_s"]
  dt_s = out["dense_dt_s"][b]
  valid = out["dense_valid"][b]
  fall_t = float(out["time_to_fall_s"][b]) if out["fell"][b] else np.inf

  # Elevation map cells in world.
  (sx, sy), res = out["grid_size"], float(out["grid_resolution"])
  elev = out["elevation_fy"][b]
  ny, nx = elev.shape
  gx = -sx / 2 + np.arange(nx) * res
  gy = -sy / 2 + np.arange(ny) * res
  cx, cy = np.meshgrid(gx, gy)
  cells_fy = np.stack([cx, cy, elev], axis=-1).reshape(-1, 3)
  cells_w = f_pos + cells_fy @ rot.T
  ok_cells = np.isfinite(cells_fy[:, 2])
  # Fixed range around the ground under the trunk at the fault (~0.3 m
  # below it): blue 0.4 m below that ground, red 0.4 m above.
  ground = -0.3
  cell_rgb = _height_rgb(cells_fy[:, 2], ground - 0.4, ground + 0.4)
  touched_ch = list(out["risk_channels"]).index("touched")
  touched = out["risk_window_fy"][b][:, touched_ch].reshape(len(edges) - 1, -1) > 0
  first_w = np.where(touched.any(0), touched.argmax(0), -1)
  expected_w = (
    f_pos
    + np.nan_to_num(
      np.concatenate(
        [out["dense_expected_fy"][b, :, :2], np.zeros((len(dt_s), 1))], axis=-1
      )
    )
    @ rot.T
  )

  # One fixed camera per trial, framing the fault pose and the whole trunk
  # path over the horizon (so the robot stays in view as it walks or falls).
  path = out["dense_base_pos_w"][b, pre:][valid[pre:]]
  pts = np.vstack([f_pos[None], path])
  lo_xy, hi_xy = pts[:, :2].min(0), pts[:, :2].max(0)
  centre = np.array([*(lo_xy + hi_xy) / 2, f_pos[2] - 0.2])
  extent = float(np.max(hi_xy - lo_xy))
  cam = mujoco.MjvCamera()
  cam.lookat[:] = centre
  cam.distance = max(3.0, 1.3 * extent + 2.0)
  cam.azimuth = np.degrees(yaw) + 140.0
  cam.elevation = -40.0
  opt = mujoco.MjvOption()

  frames = []
  for k, t_k in enumerate(edges):
    data.qpos[:] = out["frame_qpos"][b, k]
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, cam, scene_option=opt)
    scene = renderer.scene
    flat = np.eye(3)
    # Elevation map tiles.
    for i in np.flatnonzero(ok_cells):
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        [0.45 * res, 0.45 * res, 0.003],
        cells_w[i] + [0, 0, 0.006],
        rot,
        [*cell_rgb[i], 0.35],
      )
    # Impact cells touched up to this frame, coloured by first window.
    for i in np.flatnonzero((first_w >= 0) & (first_w < k)):
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        [0.48 * res, 0.48 * res, 0.03],
        cells_w[i] + [0, 0, 0.04],
        rot,
        [*WINDOW_COLORS[first_w[i] % len(WINDOW_COLORS)], 1.0],
        emission=0.8,  # Self-lit, so the colour reads the same in shadow.
      )
    # Commanded path (full horizon) and trunk path so far.
    for j in range(pre, len(dt_s), 10):
      if valid[j]:
        _add(
          scene,
          mujoco.mjtGeom.mjGEOM_SPHERE,
          [0.02, 0, 0],
          expected_w[j] + [0, 0, 0.05],
          flat,
          [0.5, 0.5, 0.5, 0.8],
        )
    for j in range(pre, len(dt_s), 3):
      if valid[j] and dt_s[j] <= t_k + 1e-6:
        color = [0.9, 0.1, 0.1, 1.0] if dt_s[j] >= fall_t else [0.1, 0.4, 1.0, 1.0]
        _add(
          scene,
          mujoco.mjtGeom.mjGEOM_SPHERE,
          [0.018, 0, 0],
          out["dense_base_pos_w"][b, j],
          flat,
          color,
        )
    # Fault pose arrow and the faulted joint(s).
    arrow = rot @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  # Local z -> heading.
    _add(
      scene,
      mujoco.mjtGeom.mjGEOM_ARROW,
      [0.025, 0.025, 0.4],
      f_pos + [0, 0, 0.2],
      arrow,
      [0, 0, 0, 1],
    )
    for name in joints:
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        [0.045, 0, 0],
        data.xanchor[jnt_ids[name]],
        flat,
        MODE_COLORS.get(mode, (1, 1, 1, 1)),
      )
    frames.append(renderer.render().copy())
  return frames


def save_strip(
  frames: list[np.ndarray], out: dict[str, np.ndarray], b: int, path: Path
) -> None:
  edges = out["window_edges_s"]
  mode = str(out["mode_names"][out["fault_mode"][b]])
  joints = "+".join(str(out["joint_names"][j]) for j in out["fault_joint"][b])
  row, col = out["terrain_tile_fault"][b]
  terrain = str(out["terrain_names"][col]) if col >= 0 else "?"
  fell = (
    f"fell at +{float(out['time_to_fall_s'][b]):.2f} s"
    if out["fell"][b]
    else "stayed up"
  )
  cols = 3
  rows = int(np.ceil(len(frames) / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(6.4 * cols, 4.9 * rows + 0.8))
  axes = np.atleast_1d(axes).flatten()
  for ax in axes:
    ax.set_axis_off()
  for k, (ax, img) in enumerate(zip(axes, frames, strict=False)):
    ax.imshow(img)
    ax.set_title("fault (+0 s)" if k == 0 else f"+{edges[k]:.0f} s", fontsize=13)
  fig.suptitle(
    f"trial {b}: {joints} {mode} on {terrain} row {row} -- {fell}\n"
    "Tiles: elevation map (blue low, red high). Raised boxes: trunk impact "
    "cells by second after the fault (yellow 0-1 s, orange 1-2, red 2-3, "
    "magenta 3-4, purple 4-5).\nDots: trunk path (blue up, red after fall); "
    "grey: commanded path. Black arrow: trunk pose at the fault. Sphere: "
    "failed joint (red lock, blue free).",
    fontsize=11,
  )
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def pick_trials(out: dict[str, np.ndarray], per_mode: int) -> list[int]:
  """Up to ``per_mode`` kept trials per fault mode: fallen ones first (by
  most trunk impact cells), then survivors."""
  picks = []
  kept = out["env_kept"]
  touched_ch = list(out["risk_channels"]).index("touched")
  cells = (out["risk_fy"][:, touched_ch] > 0).sum(axis=(1, 2))
  for m in np.unique(out["fault_mode"][kept]):
    sel = np.flatnonzero(kept & (out["fault_mode"] == m))
    fell = sel[out["fell"][sel]]
    up = sel[~out["fell"][sel]]
    order = list(fell[np.argsort(-cells[fell])]) + list(up)
    n_fell = min(len(fell), max(per_mode - 1, 1))
    picks += order[:n_fell] + list(up[: per_mode - n_fell])
  return [int(b) for b in picks]


def render_trials(
  model: mujoco.MjModel,
  out: dict[str, np.ndarray],
  per_mode: int,
  out_dir: Path,
  size: tuple[int, int] = (480, 640),
) -> list[Path]:
  """Render ``pick_trials`` and save one strip PNG per trial."""
  out_dir.mkdir(parents=True, exist_ok=True)
  h, w = size
  model.vis.global_.offheight = max(model.vis.global_.offheight, h)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
  jnt_ids = {
    str(n): mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
    for n in out["joint_names"]
  }
  missing = [n for n, i in jnt_ids.items() if i < 0]
  if missing:
    raise ValueError(f"Joints not found in the model: {missing}")
  renderer = mujoco.Renderer(model, h, w)
  paths = []
  try:
    for b in pick_trials(out, per_mode):
      frames = render_trial(model, renderer, out, b, jnt_ids)
      mode = str(out["mode_names"][out["fault_mode"][b]])
      path = out_dir / f"trial{b:03d}_{mode}.png"
      save_strip(frames, out, b, path)
      paths.append(path)
  finally:
    renderer.close()
  return paths
