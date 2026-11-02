"""Replay one trial of a nominal run (``failbench/scripts/nominal.py``) as a
video: a kinematic replay of the logged base pose and joint angles on the
regenerated terrain, as ``play_trial.py``. Drawn into the scene: the trial's
map (translucent purple: the planner's costmap only, not the physics), its
stretch of global path (green), the subgoal (green pillar) and the trunk's
path so far (blue). The caption gives the map layout, the planner, the time,
the command and the measured velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw

from failbench.env_cfg import failbench_env_cfg
from failbench.maps import LAYOUTS
from failbench.scripts.play_trial import _BLUE, _GREEN, _PURPLE, _add, _line, _yaw
from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class PlayNominalConfig:
  log: str
  trial: int
  output: str
  size: tuple[int, int] = (540, 960)
  distance: float = 5.0
  elevation_deg: float = -50.0
  azimuth_offset_deg: float = 180.0
  """Camera azimuth relative to the trunk's starting heading."""
  frame_every: int = 2
  after_reach_s: float = 2.0
  """Stop this long after the robot reaches its subgoal."""
  device: str = "cpu"


def main():
  cfg = tyro.cli(PlayNominalConfig)
  z = np.load(cfg.log)
  b = cfg.trial
  terrain, seed = str(z["terrain"]), int(z["terrain_seed"])
  env_cfg = failbench_env_cfg(terrain=terrain, terrain_seed=None if seed < 0 else seed)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  model = env.sim.mj_model
  h, w = cfg.size
  model.vis.global_.offheight = max(model.vis.global_.offheight, h)
  model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
  data = mujoco.MjData(model)
  jnt = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")
    for n in z["joint_names"]
  ]
  base = next(
    i
    for i in range(model.njnt)
    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
    and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i).startswith("robot/")
  )
  base_adr = model.jnt_qposadr[base]
  q_adr = [model.jnt_qposadr[j] for j in jnt]

  pos, quat, dt = z["base_pos_w"][b], z["base_quat_w"][b], float(z["step_dt"])
  real = int(np.ceil(int(z["path_real"][b]) / int(z["path_every"])))
  path = z["path"][b, : max(real, 2)]
  goal = z["subgoal"][b]
  boxes = z["boxes"][b]
  boxes = boxes[boxes[:, 2] > 0]
  ground = pos[0, 2] - 0.32
  layout, planner = LAYOUTS[int(z["layout"][b])], str(z["planner"])
  reached = float(z["reached_s"][b])
  steps = min(len(pos), int(z["episode_steps"][b]))
  if np.isfinite(reached):
    steps = min(steps, round((reached + cfg.after_reach_s) / dt))

  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.distance = cfg.distance
  cam.elevation = cfg.elevation_deg
  cam.azimuth = math.degrees(_yaw(quat[0])) + cfg.azimuth_offset_deg
  renderer = mujoco.Renderer(model, h, w)
  lookat = pos[0].copy()
  frames = []
  for t in range(0, steps, cfg.frame_every):
    data.qpos[base_adr : base_adr + 3] = pos[t]
    data.qpos[base_adr + 3 : base_adr + 7] = quat[t]
    data.qpos[q_adr] = z["joint_pos"][b, t]
    mujoco.mj_forward(model, data)
    lookat += 0.05 * (pos[t] - lookat)
    cam.lookat[:] = lookat
    renderer.update_scene(data, cam)
    scene = renderer.scene
    _add(
      scene,
      mujoco.mjtGeom.mjGEOM_CYLINDER,
      (0.08, 0.4, 0),
      (goal[0], goal[1], ground + 0.4),
      _GREEN,
    )
    for cx, cy, hx, hy, yaw in boxes:
      c, s = math.cos(yaw), math.sin(yaw)
      mat = np.array([c, -s, 0, s, c, 0, 0, 0, 1], dtype=np.float64)
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        (hx, hy, 0.4),
        (cx, cy, ground + 0.4),
        _PURPLE,
        mat,
      )
    for a, c in zip(path[:-1], path[1:], strict=True):
      _line(scene, (*a, ground + 0.03), (*c, ground + 0.03), 0.015, _GREEN)
    k = np.arange(0, t + 1, 5)
    for a, c in zip(k[:-1], k[1:], strict=True):
      _line(scene, pos[a], pos[c], 0.012, _BLUE)
    img = Image.fromarray(renderer.render())
    draw = ImageDraw.Draw(img)
    cmd, v, wz = z["cmd"][b, t], z["lin_vel_b"][b, t], z["ang_vel_b"][b, t, 2]
    lines = [
      f"{terrain} seed {seed}, trial {b}: {layout} map, {planner} tracking its global path",
      f"t = {t * dt:.1f} s"
      + (f" (reached the subgoal at {reached:.1f} s)" if np.isfinite(reached) else ""),
      f"cmd  vx {cmd[0]:+.2f}  vy {cmd[1]:+.2f}  wz {cmd[2]:+.2f}",
      f"real vx {v[0]:+.2f}  vy {v[1]:+.2f}  wz {wz:+.2f}",
    ]
    draw.rectangle((8, 8, 560, 18 + 18 * len(lines)), fill=(0, 0, 0))
    for i, line in enumerate(lines):
      draw.text((16, 14 + 18 * i), line, fill=(255, 255, 255))
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
