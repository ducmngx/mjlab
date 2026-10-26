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
- the faulted joint once the fault fires (red = locked, blue = weak);
- with a fault-risk planner (``--mppi.fault-risk.p`` > 0), while the robot
  is healthy: where each priced fault would throw it if it fired at points
  along the plan (one colour per fault, warm = limp, cool = seized; the
  ``risk_fire`` point drawn bold with its 2 sigma ring), and a panel with
  each fault's share of the plan's fault risk and P(fall) once fired.

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
from failbench.mppi.planner import FaultRiskCfg
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
  look_ahead_m: float = 0.0
  """Aim the camera this far ahead of the trunk along its heading."""
  risk_full: float = 0.3
  """Risk at which the predicted path is drawn fully red."""
  contact_min: float = 0.15
  """Contact probability below which a cell isn't drawn."""
  max_contact_cells: int = 1500
  risk_fire: int = 1
  """Firing point (index into ``fire_s``) whose throws are drawn bold."""
  device: str = "cpu"


# Fault-risk branches: warm for a limp joint, cool for a seized one.
_LIMP = ((0.95, 0.3, 0.1), (1.0, 0.65, 0.0), (0.85, 0.1, 0.45), (0.95, 0.85, 0.2))
_SEIZED = ((0.1, 0.45, 0.95), (0.0, 0.75, 0.75), (0.45, 0.3, 0.95), (0.3, 0.8, 0.3))


def _branch_colours(names: list[str]) -> list[tuple[float, float, float]]:
  limp = iter(_LIMP * 3)
  seized = iter(_SEIZED * 3)
  return [next(limp) if n.endswith(":free") else next(seized) for n in names]


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
  risk_names = [str(n) for n in log["risk_faults"]] if "risk_faults" in log else []
  risk_rgb = _branch_colours(risk_names)
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
  waypoints = (
    log["waypoints_w"][b] if "waypoints_w" in log.files else log["goal_w"][b : b + 1]
  )
  goal = waypoints[-1]
  down = fallen(torch.as_tensor(dense["tilt"]), torch.as_tensor(dense["base_height"]))
  ever_down = np.maximum.accumulate(down.numpy())
  cfg_fire_s = list(FaultRiskCfg().fire_s)

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
    ahead = cfg.look_ahead_m * np.array(
      [math.cos(_yaw(quat[t])), math.sin(_yaw(quat[t])), 0]
    )
    lookat += 0.05 * (pos[t] + ahead - lookat)
    if t % cfg.frame_every:
      continue
    cam.lookat[:] = lookat
    renderer.update_scene(data, cam)
    scene = renderer.scene
    ground = pos[t, 2] - dense["base_height"][t]
    for wp in waypoints[:-1]:  # intermediate waypoints: short pillars
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
        (0.06, 0.15, 0),
        (wp[0], wp[1], ground + 0.15),
        _GREEN,
      )
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
    risky = (
      planned
      and risk_names
      and not dense["faulted"][t]
      and not np.isnan(float(cyc["risk_branch"][c, 0]))
    )
    if risky:
      z = ground + 0.06
      thrown = cyc["risk_thrown_w"][c].astype(np.float32)  # [F, B, Tp, 2]
      for f in range(thrown.shape[0]):
        bold = f == cfg.risk_fire
        for i, rgb in enumerate(risk_rgb):
          pts = thrown[f, i]
          pts = np.concatenate([pts, np.full((len(pts), 1), z)], -1)
          rgba = (*rgb, 0.95 if bold else 0.3)
          for a, e in zip(pts[:-1:2], pts[2::2], strict=False):
            _line(scene, a, e, 0.012 if bold else 0.005, rgba)
          if bold:
            r = 2 * float(cyc["risk_thrown_std"][c, f, i, -1])
            _add(
              scene,
              mujoco.mjtGeom.mjGEOM_CYLINDER,
              (max(r, 0.02), 0.004, 0),
              (*pts[-1, :2], z),
              (*rgb, 0.1),
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
    if risky:
      share = cyc["risk_branch"][c].astype(np.float32)
      fall = cyc["risk_p_fall"][c, cfg.risk_fire].astype(np.float32)
      x0, y0 = w - 350, 8
      draw.rectangle((x0, y0, w - 8, y0 + 34 + 18 * len(risk_names)), fill=(0, 0, 0))
      fire_s = cfg_fire_s[cfg.risk_fire] if cfg.risk_fire < len(cfg_fire_s) else 0
      draw.text(
        (x0 + 8, y0 + 6),
        f"fault risk {share.sum():.2f} (bold: fired at {fire_s:g} s)",
        fill=(255, 255, 255),
      )
      top = max(float(share.max()), 1e-6)
      for i, (name, rgb) in enumerate(zip(risk_names, risk_rgb, strict=True)):
        y = y0 + 26 + 18 * i
        col = tuple(round(255 * v) for v in rgb)
        draw.rectangle((x0 + 8, y + 3, x0 + 18, y + 13), fill=col)
        joint_, mode = name.split(":")
        label = (
          f"{joint_.removesuffix('_joint')} {'limp' if mode == 'free' else 'lock'}"
        )
        draw.text((x0 + 24, y), f"{label:13} P(fall) {fall[i]:.2f}", fill=col)
        bar = round(70 * float(share[i]) / top)
        draw.rectangle((x0 + 232, y + 3, x0 + 232 + bar, y + 13), fill=col)
        draw.text((x0 + 236 + bar, y), f"{share[i]:.2f}", fill=(220, 220, 220))
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
