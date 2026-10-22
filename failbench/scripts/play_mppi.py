"""Replay one robot of a ``run_mppi.py`` log as a video.

A kinematic replay, as ``play_trial.py``: the terrain is regenerated from
its preset and seed and the robot posed from the logged base pose and
joint angles. Drawn into the scene:

- the goal (green pillar) and the trunk's path so far (blue, red once
  fallen);
- MPPI's current plan: its commanded path (green) and the path the
  dynamics model predicts for it, coloured by risk at each step (orange
  when safe, red as P(fallen by then) or P(trunk contact then) reaches
  ``risk_full``), with the predicted 2 sigma spread every second (orange
  rings) and a ghost trunk where the robot is predicted to be at 5 s; the
  gap between the green and predicted paths is the predicted drift;
- for a fault-blind planner (``run_mppi.py --no-fault-aware``), the same
  plan as the model predicts it given the true fault (purple path and
  ghost trunk);
- where the model predicts a non-foot body part touches the ground over
  the plan's 5 s (magenta cells, more opaque = likelier);
- the faulted joint once the fault fires (red = locked, blue = weak).

The caption gives the time, the fault, the command, the plan's predicted
drift at 5 s, P(fall in 5 s), highest P(trunk contact) and cost terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
import tyro
from PIL import Image, ImageDraw

from failbench.data.build import fallen
from failbench.env_cfg import failbench_env_cfg
from failbench.faults import LOCK, TORQUE, FaultFamily, faulted_joint
from failbench.scripts.play_trial import _BLUE, _GREEN, _ORANGE, _RED, _add, _line, _yaw
from mjlab.envs import ManagerBasedRlEnv

_PLAN = (0.15, 0.8, 0.2, 0.95)
_PURPLE = (0.55, 0.2, 0.95, 0.95)


@dataclass(frozen=True)
class PlayMppiConfig:
  log: str
  """``run_mppi.py`` output (``.npz``)."""
  env: int
  output: str
  size: tuple[int, int] = (540, 960)
  distance: float = 4.0
  elevation_deg: float = -35.0
  azimuth_offset_deg: float = 140.0
  frame_every: int = 2
  """Render every n-th control step."""
  risk_full: float = 0.3
  """Risk at which the predicted path is drawn fully red."""
  contact_min: float = 0.15
  """Contact probability below which a cell isn't drawn."""
  max_contact_cells: int = 1500
  device: str = "cpu"


def main():
  cfg = tyro.cli(PlayMppiConfig)
  log = np.load(cfg.log)
  b = cfg.env
  terrain, seed = str(log["terrain"]), int(log["terrain_seed"])
  env_cfg = failbench_env_cfg(terrain=terrain, terrain_seed=None if seed < 0 else seed)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  model = env.sim.mj_model
  h, w = cfg.size
  model.vis.global_.offheight = max(model.vis.global_.offheight, h)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
  data = mujoco.MjData(model)

  joint_names = [str(n) for n in log["joint_names"]]
  jnt = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
    for n in joint_names
  ]
  base = next(
    i
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).startswith("robot/")
  )
  base_adr = model.jnt_qposadr[base]
  q_adr = [model.jnt_qposadr[j] for j in jnt]

  dense = {k[6:]: log[k][b] for k in log.files if k.startswith("dense_")}
  cycle_step = log["cycle_step"]
  cyc = {
    k[6:]: log[k][b] for k in log.files if k.startswith("cycle_") and k != "cycle_step"
  }
  contact_prob = log["contact_prob"][b] if "contact_prob" in log.files else None
  contact_height = log["contact_height"][b] if "contact_prob" in log.files else None
  cells = 0 if contact_prob is None else contact_prob.shape[-1]
  res = 0.05
  dt = float(log["step_dt"])
  pos, quat = dense["base_pos_w"], dense["base_quat_w"]
  params = log["fault_params"][b]
  family = FaultFamily(int(log["fault_family"][b])).name
  joint = int(faulted_joint(torch.as_tensor(params)))
  what = family.lower()
  if joint >= 0:
    what += f" {joint_names[joint]}"
    if family == "WEAK":
      what += f" torque x{params[joint, TORQUE]:.2f}"
  fault_t = int(log["fault_step"][b]) * dt
  goal = log["goal_w"][b]
  down = fallen(torch.as_tensor(dense["tilt"]), torch.as_tensor(dense["base_height"]))
  ever_down = np.maximum.accumulate(down.numpy())

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance = cfg.distance
  cam.elevation = cfg.elevation_deg
  cam.azimuth = math.degrees(_yaw(quat[0])) + cfg.azimuth_offset_deg
  renderer = mujoco.Renderer(model, h, w)
  lookat = pos[0].copy()
  frames = []
  # Planning cycles that ran, and the latest one at or before each step.
  ran = np.flatnonzero(cycle_step >= 0)
  for t in range(len(pos)):
    j = np.searchsorted(cycle_step[ran], t, side="right") - 1
    c = int(ran[j]) if j >= 0 else -1
    data.qpos[base_adr : base_adr + 3] = pos[t]
    data.qpos[base_adr + 3 : base_adr + 7] = quat[t]
    data.qpos[q_adr] = dense["joint_pos"][t]
    mujoco.mj_forward(model, data)
    lookat += 0.05 * (pos[t] - lookat)
    if t % cfg.frame_every:
      continue
    cam.lookat[:] = lookat
    renderer.update_scene(data, cam)
    scene = renderer.scene
    ground = pos[t, 2] - dense["base_height"][t]
    _add(
      scene,
      mujoco.mjtGeom.mjGEOM_CYLINDER,
      (0.08, 0.4, 0),
      (goal[0], goal[1], ground + 0.4),
      _GREEN,
    )
    k = np.arange(0, t + 1, 5)
    for a, e in zip(k[:-1], k[1:], strict=True):
      _line(scene, pos[a], pos[e], 0.012, _RED if ever_down[e] else _BLUE)
    planned = c >= 0 and not np.isnan(cyc["path_w"][c, 0, 0])
    blind = False
    if planned:
      z = ground + 0.05
      risk = np.maximum(cyc["pred_p_fallen"][c], cyc["pred_p_collision"][c])
      for key in ("path_w", "pred_w"):
        pts = np.concatenate([pos[t : t + 1, :2], cyc[key][c]], axis=0)
        pts = np.concatenate([pts, np.full((len(pts), 1), z)], -1)
        for i, (a, e) in enumerate(zip(pts[:-1], pts[1:], strict=True)):
          if key == "path_w":
            _line(scene, a, e, 0.01, _PLAN)
            continue
          u = float(np.clip(risk[i] / cfg.risk_full, 0.0, 1.0))
          rgba = tuple(
            (1 - u) * np.array(_ORANGE) + u * np.array((0.9, 0.05, 0.05, 0.95))
          )
          _line(scene, a, e, 0.012 + 0.01 * u, rgba)
      # Ghost trunk at the predicted 5 s pose.
      end, yaw_end = cyc["pred_w"][c, -1], float(cyc["pred_yaw_w"][c, -1])
      cy, sy = math.cos(yaw_end), math.sin(yaw_end)
      rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        (0.35, 0.15, 0.06),
        (end[0], end[1], cyc["origin_w"][c, 2]),
        (1.0, 0.6, 0.0, 0.35),
        rot,
      )
      # A fault-blind planner's plan as the model predicts it given the
      # true fault: path and ghost trunk in purple.
      blind = "true_pred_w" in cyc and not np.allclose(
        cyc["true_pred_w"][c], cyc["pred_w"][c], equal_nan=True
      )
      if blind:
        pts = np.concatenate([pos[t : t + 1, :2], cyc["true_pred_w"][c]], axis=0)
        pts = np.concatenate([pts, np.full((len(pts), 1), z + 0.01)], -1)
        for a, e in zip(pts[:-1], pts[1:], strict=True):
          _line(scene, a, e, 0.014, _PURPLE)
        tend = cyc["true_pred_w"][c, -1]
        # Heading from the path's last step (yaw isn't logged for it).
        step_ = tend - cyc["true_pred_w"][c, -2]
        trot = rot
        if np.hypot(*step_) > 0.01:
          th = math.atan2(step_[1], step_[0])
          ct, st = math.cos(th), math.sin(th)
          trot = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
        _add(
          scene,
          mujoco.mjtGeom.mjGEOM_BOX,
          (0.35, 0.15, 0.06),
          (tend[0], tend[1], cyc["origin_w"][c, 2]),
          (0.55, 0.2, 0.95, 0.35),
          trot,
        )
      # Predicted body contact over the next 5 s.
      if contact_prob is not None and contact_height is not None:
        p = contact_prob[c].max(axis=0) / 255.0
        hgt = contact_height[c].astype(np.float32)
        iy, ix = np.nonzero((p >= cfg.contact_min) & ~np.isnan(hgt))
        order = np.argsort(-p[iy, ix])[: cfg.max_contact_cells]
        ox, oy, oz, oyaw = cyc["origin_w"][c]
        co, so = math.cos(oyaw), math.sin(oyaw)
        for k_ in order:
          lx = (ix[k_] - (cells - 1) / 2) * res
          ly = (iy[k_] - (cells - 1) / 2) * res
          q = float(p[iy[k_], ix[k_]])
          _add(
            scene,
            mujoco.mjtGeom.mjGEOM_BOX,
            (res / 2, res / 2, 0.004),
            (
              ox + co * lx - so * ly,
              oy + so * lx + co * ly,
              oz + hgt[iy[k_], ix[k_]] + 0.01,
            ),
            (0.85, 0.1, 0.85, 0.2 + 0.7 * q),
          )
      for s in range(9, len(cyc["pred_w"][c]), 10):
        r = 2 * float(np.linalg.norm(cyc["pred_std"][c, s, :2]))
        p = cyc["pred_w"][c, s]
        _add(
          scene,
          mujoco.mjtGeom.mjGEOM_CYLINDER,
          (max(r, 0.02), 0.005, 0),
          (p[0], p[1], z),
          (1.0, 0.6, 0.0, 0.35),
        )
    if dense["faulted"][t] and joint >= 0:
      rgba = _RED if params[joint, LOCK] > 0.5 else _BLUE
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (0.045, 0, 0),
        data.xanchor[jnt[joint]],
        rgba,
      )

    img = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(img)
    cmd = dense["cmd"][t]
    if dense["faulted"][t] and joint >= 0:
      when = f"FAULT {what} at {fault_t:.1f} s"
    else:
      when = f"healthy (fault: {what})" if joint >= 0 else "healthy (no fault)"
    dist = float(np.linalg.norm(goal[:2] - pos[t, :2]))
    lines = [
      f"{terrain}, env {b}   t = {t * dt:5.2f} s   goal {dist:.1f} m",
      when,
      f"cmd  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
    ]
    if planned:
      d = cyc["drift_end"][c]
      lines.append(
        f"predicted drift at 5 s: dx {d[0]:+.2f} dy {d[1]:+.2f} m"
        f" ({np.hypot(d[0], d[1]):.2f} m), yaw {math.degrees(d[2]):+.0f} deg"
      )
      if blind:
        dt_ = cyc["true_pred_w"][c, -1] - cyc["path_w"][c, -1]
        lines.append(
          f"with the true fault (purple): drift {np.hypot(*dt_):.2f} m,"
          f" P(fall in 5 s) {cyc['true_p_fallen'][c]:.2f}"
        )
      lines.append(
        f"P(fall in 5 s) {cyc['p_fallen'][c]:.2f}, highest P(trunk contact)"
        f" {np.nanmax(cyc['pred_p_collision'][c]):.2f}"
      )
      costs = [
        f"{k[5:]} {cyc[k][c]:.2f}"
        for k in cyc
        if k.startswith("cost_") and not np.isnan(cyc[k][c])
      ]
      lines.append("cost " + ", ".join(costs))
    else:
      lines.append("goal follower (MPPI not started)" if not ever_down[t] else "")
    status = "FALLEN" if ever_down[t] else "upright"
    if dense["trunk_contact"][t]:
      status += ", trunk/head contact"
    lines.append(status)
    draw.rectangle((8, 8, 640, 18 + 18 * len(lines)), fill=(0, 0, 0))
    for i, line in enumerate(lines):
      hot = i == 1 and dense["faulted"][t] and joint >= 0
      fill = (255, 90, 90) if hot else (255, 255, 255)
      draw.text((16, 14 + 18 * i), line, fill=fill)
    frames.append(np.asarray(img))
  renderer.close()
  env.close()
  out = Path(cfg.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(
    out, frames, fps=round(1 / (dt * cfg.frame_every)), macro_block_size=1
  )
  print(f"[INFO] {len(frames)} frames -> {out}")


if __name__ == "__main__":
  main()
