"""Replay a logged FailBench trial as a video.

The trial's terrain is regenerated from its preset and seed (the generator
is deterministic), and the robot is posed every control step from the raw
shard's base pose and joint angles: a kinematic replay of exactly what was
logged, not a re-simulation. A chase camera follows the trunk. Drawn into
the scene:

- the goal (green pillar);
- the trunk's path so far (blue, red once fallen);
- after the fault, the post-fault plan's commanded path, the logged
  command integrated from the fault pose (orange), and a red line from the
  trunk to where the plan says it should be now (the drift);
- the faulted joint (red = locked, blue = weak).

A caption gives the time, the plan (goal follower before the fault, then
hold or the current segment and its target twist), the command, the
measured velocity, the drift and the labels.
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
from mjlab.envs import ManagerBasedRlEnv

_BLUE = (0.15, 0.45, 1.0, 0.9)
_RED = (1.0, 0.15, 0.1, 0.9)
_ORANGE = (1.0, 0.6, 0.0, 0.9)
_GREEN = (0.1, 0.85, 0.3, 0.8)


@dataclass(frozen=True)
class PlayConfig:
  raw: str
  """Raw shard (``generate.py`` output)."""
  trial: int
  """Trial (env) index in the shard."""
  output: str
  size: tuple[int, int] = (540, 960)
  """Video (height, width)."""
  distance: float = 3.2
  elevation_deg: float = -30.0
  azimuth_offset_deg: float = 140.0
  """Camera azimuth relative to the trunk heading at the fault."""
  device: str = "cpu"


def _yaw(quat: np.ndarray) -> float:
  w, x, y, z = quat
  return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _add(scene: mujoco.MjvScene, kind, size, pos, rgba, mat=None) -> None:
  if scene.ngeom >= scene.maxgeom:
    return
  mujoco.mjv_initGeom(
    scene.geoms[scene.ngeom],
    kind,
    np.asarray(size, dtype=np.float64),
    np.asarray(pos, dtype=np.float64),
    (np.eye(3) if mat is None else mat).flatten(),
    np.asarray(rgba, dtype=np.float32),
  )
  scene.ngeom += 1


def _line(scene: mujoco.MjvScene, a, b, width: float, rgba) -> None:
  if scene.ngeom >= scene.maxgeom:
    return
  g = scene.geoms[scene.ngeom]
  mujoco.mjv_initGeom(
    g,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
    np.zeros(3),
    np.zeros(3),
    np.eye(3).flatten(),
    np.asarray(rgba, dtype=np.float32),
  )
  mujoco.mjv_connector(
    g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.asarray(a, float), np.asarray(b, float)
  )
  scene.ngeom += 1


def _cmd_path(cmd: np.ndarray, x0: np.ndarray, yaw0: float, dt: float) -> np.ndarray:
  """World (x, y) after each step of body twists ``cmd [T, 3]`` from
  ``x0``, heading ``yaw0``."""
  yaw = yaw0 + np.concatenate([[0.0], np.cumsum(cmd[:-1, 2] * dt)])
  c, s = np.cos(yaw), np.sin(yaw)
  d = np.stack([c * cmd[:, 0] - s * cmd[:, 1], s * cmd[:, 0] + c * cmd[:, 1]], -1)
  return x0[None, :2] + np.cumsum(d * dt, axis=0)


def main():
  cfg = tyro.cli(PlayConfig)
  raw = np.load(cfg.raw)
  b = cfg.trial
  terrain, seed = str(raw["terrain"]), int(raw["terrain_seed"])
  env_cfg = failbench_env_cfg(terrain=terrain, terrain_seed=None if seed < 0 else seed)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  model = env.sim.mj_model
  h, w = cfg.size
  model.vis.global_.offheight = max(model.vis.global_.offheight, h)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
  data = mujoco.MjData(model)

  joint_names = [str(n) for n in raw["joint_names"]]
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

  dense = {k[6:]: raw[k][b] for k in raw.files if k.startswith("dense_")}
  pre, dt = int(raw["pre_steps"]), float(raw["step_dt"])
  pos, quat = dense["base_pos_w"], dense["base_quat_w"]
  params = raw["fault_params"][b]
  family = FaultFamily(int(raw["fault_family"][b])).name
  joint = int(faulted_joint(torch.as_tensor(params)))
  what = family.lower()
  if joint >= 0:
    what += f" {joint_names[joint]}"
    if family == "WEAK":
      what += f" torque x{params[joint, TORQUE]:.2f}"
  hold = bool(raw["plan_hold"][b])
  goal = raw["goal_w"][b]
  down = fallen(torch.as_tensor(dense["tilt"]), torch.as_tensor(dense["base_height"]))
  ever_down = np.maximum.accumulate(down.numpy() & (np.arange(len(down)) >= pre))
  # The command on step t is applied during step t + 1.
  plan_xy = _cmd_path(dense["cmd"][pre:-1], pos[pre], _yaw(quat[pre]), dt)
  plan_xy = np.concatenate([pos[pre : pre + 1, :2], plan_xy])  # [T - pre, 2]

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance = cfg.distance
  cam.elevation = cfg.elevation_deg
  cam.azimuth = math.degrees(_yaw(quat[pre])) + cfg.azimuth_offset_deg
  renderer = mujoco.Renderer(model, h, w)
  lookat = pos[0].copy()
  frames = []
  for t in np.flatnonzero(dense["valid"]):
    data.qpos[base_adr : base_adr + 3] = pos[t]
    data.qpos[base_adr + 3 : base_adr + 7] = quat[t]
    data.qpos[q_adr] = dense["joint_pos"][t]
    mujoco.mj_forward(model, data)
    lookat += 0.1 * (pos[t] - lookat)
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
    for a, c in zip(k[:-1], k[1:], strict=True):
      _line(scene, pos[a], pos[c], 0.012, _RED if ever_down[c] else _BLUE)
    rel = t - pre
    if rel >= 0:
      z = pos[pre, 2]
      pts = np.concatenate([plan_xy[::5], np.full((len(plan_xy[::5]), 1), z)], -1)
      for a, c in zip(pts[:-1], pts[1:], strict=True):
        _line(scene, a, c, 0.012, _ORANGE)
      now = np.array([*plan_xy[rel], z])
      _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), now, _ORANGE)
      _line(scene, pos[t], now, 0.008, _RED)
      if joint >= 0:
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
    cmd, v, wz = dense["cmd"][t], dense["lin_vel_b"][t], dense["ang_vel_b"][t, 2]
    if rel < 0:
      plan = "goal follower (closed loop)"
    elif hold:
      plan = "plan: HOLD the fault-time command"
    else:
      tg = dense["plan_target"][t]
      plan = (
        f"plan: segment {int(dense['plan_seg'][t])}, target"
        f" vx {tg[0]:+.2f} vy {tg[1]:+.2f} wz {tg[2]:+.2f}"
      )
    status = "FALLEN" if ever_down[t] else "upright"
    if dense["trunk_contact"][t]:
      status += ", trunk/head contact"
    lines = [
      f"{terrain} seed {seed}, trial {b}: {what}",
      f"t = {rel * dt:+.2f} s {'(before fault)' if rel < 0 else 'after fault'}",
      plan,
      f"cmd  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
      f"real vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {wz:+.2f}",
    ]
    if rel >= 0:
      lines.append(f"drift from plan {np.linalg.norm(pos[t, :2] - plan_xy[rel]):.2f} m")
    lines.append(f"{status}, trunk {dense['base_height'][t]:.2f} m above ground")
    draw.rectangle((8, 8, 520, 18 + 18 * len(lines)), fill=(0, 0, 0))
    for i, line in enumerate(lines):
      fill = (255, 90, 90) if i == 1 and rel >= 0 else (255, 255, 255)
      draw.text((16, 14 + 18 * i), line, fill=fill)
    frames.append(np.asarray(img))
  renderer.close()
  env.close()
  out = Path(cfg.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimwrite(out, frames, fps=round(1 / dt), macro_block_size=1)
  print(f"[INFO] {len(frames)} frames -> {out}")


if __name__ == "__main__":
  main()
