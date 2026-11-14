"""A MuJoCo window on one robot of a ``run_mppi.py --live`` run, paced to the
wall clock: the robot, the plan it runs (green), the last plan computed
(yellow, until it lands) and the global path (blue), with the planner's
compute time and rate in the corner."""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np


def _segments(scn: mujoco.MjvScene, pts: np.ndarray, z: float, rgba, width: float):
  for a, b in zip(pts[:-1], pts[1:], strict=False):
    if scn.ngeom >= scn.maxgeom:
      return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
      g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba
    )
    mujoco.mjv_connector(
      g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array([*a, z]), np.array([*b, z])
    )
    scn.ngeom += 1


# The paper's environment figure's colours and light.
CLASS_HEX = {
  "person": "#eb6834",
  "glass": "#2a78d6",
  "box": "#1baf7a",
  "wall": "#52514e",
}
ROUGH_HEX = "#c9b78f"  # Rough ground: muted sand.
# Floor at 0 m and 1 m up: a warm mid grey, not white, so the scene
# prints without glare and the objects stand out.
FLOOR_LOW, FLOOR_HIGH = "#b8b2a6", "#8a857b"
BORDER_HEX = "#9e9a91"  # The ground round the tile.


def _rgba(h: str, a: float = 1.0) -> list[float]:
  h = h.lstrip("#")
  return [int(h[k : k + 2], 16) / 255 for k in (0, 2, 4)] + [a]


def _paper_style(model: mujoco.MjModel, brightness: float) -> None:
  """Colour the scene as the environment figure does (visual only): objects
  by class, the floor light grey darkening with height, rough ground sand;
  soft light, a strong ambient headlight and the scene's lights dimmed,
  all scaled by ``brightness``."""
  model.vis.headlight.ambient[:] = 0.42 * brightness
  model.vis.headlight.diffuse[:] = 0.45 * brightness
  model.vis.headlight.specular[:] = 0.05 * brightness
  model.light_diffuse[:] *= 0.35 * brightness
  model.light_specular[:] = 0.0
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  low, high = np.array(_rgba(FLOOR_LOW)), np.array(_rgba(FLOOR_HIGH))
  for g in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if name.startswith("terrain"):
      model.geom_rgba[g] = _rgba(BORDER_HEX)
    if not name.startswith("course_") or model.geom_rgba[g][3] < 0.01:
      continue  # Hidden boxes stay hidden: a decor figure is drawn instead.
    cls = name[len("course_") :].rsplit("_", 1)[0]
    if cls in CLASS_HEX:
      model.geom_rgba[g] = _rgba(CLASS_HEX[cls], 0.8 if cls == "glass" else 1.0)
    elif cls == "floor":
      if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_HFIELD:
        h = model.geom_dataid[g]
        a, n = model.hfield_adr[h], model.hfield_nrow[h] * model.hfield_ncol[h]
        rough = np.ptp(model.hfield_data[a : a + n]) > 1e-3
        model.geom_rgba[g] = _rgba(ROUGH_HEX) if rough else _rgba(FLOOR_LOW)
      else:
        top = data.geom_xpos[g][2] + model.geom_size[g][2]
        f = float(np.clip(top / 1.0, 0.0, 1.0))
        model.geom_rgba[g] = (1 - f) * low + f * high


class LiveView:
  """``update`` after each sim step; ``set_paths`` when a plan is made."""

  def __init__(self, model: mujoco.MjModel, env_idx: int = 0, brightness: float = 1.0):
    self.model = model
    _paper_style(model, brightness)
    self.data = mujoco.MjData(model)
    self.env_idx = env_idx
    self.viewer = mujoco.viewer.launch_passive(
      model, self.data, show_left_ui=False, show_right_ui=False
    )
    # The robot's trunk: the first body named like a base with a free joint.
    names = [model.body(b).name for b in range(model.nbody)]
    free = {
      int(model.jnt_bodyid[j])
      for j in range(model.njnt)
      if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    }
    self.body = next(
      (b for b in sorted(free) if any(k in names[b] for k in ("base", "trunk"))),
      min(free) if free else 0,
    )
    self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    self.viewer.cam.distance = 5.0
    self.viewer.cam.elevation = -55.0
    self.viewer.cam.azimuth = 90.0
    self.paths: dict[str, tuple[np.ndarray, tuple]] = {}
    self.cells: tuple[np.ndarray, np.ndarray, float] | None = None
    self.text = ""
    self.wall0: float | None = None
    self.sim_t = 0.0

  def running(self) -> bool:
    return self.viewer.is_running()

  def set_path(self, name: str, pts: np.ndarray | None, rgba) -> None:
    if pts is None:
      self.paths.pop(name, None)
    else:
      self.paths[name] = (np.asarray(pts, np.float64), rgba)

  def set_cells(
    self, pos: np.ndarray | None, rgba: np.ndarray | None = None, half: float = 0.0
  ) -> None:
    """Flat square tiles ``half`` (m) wide at world points ``pos [N, 3]``,
    coloured ``rgba [N, 4]`` (None: none)."""
    self.cells = None if pos is None or rgba is None else (pos, rgba, half)

  def update(self, sim_data, sim_t: float, every: int = 1, step: int = 0) -> None:
    """Show the sim at ``sim_t`` (s), sleeping so it runs no faster than the
    wall clock; drawn every ``every`` steps."""
    if self.wall0 is None:
      self.wall0 = time.perf_counter() - sim_t
    ahead = self.wall0 + sim_t - time.perf_counter()
    if ahead > 0:
      time.sleep(ahead)
    elif ahead < -0.5:
      self.wall0 -= ahead  # Fell behind (a slow cycle): don't race to catch up.
    if step % every or not self.viewer.is_running():
      return
    i = self.env_idx
    self.data.qpos[:] = sim_data.qpos[i].cpu().numpy()
    self.data.qvel[:] = sim_data.qvel[i].cpu().numpy()
    if self.model.nmocap:
      self.data.mocap_pos[:] = sim_data.mocap_pos[i].cpu().numpy()
      self.data.mocap_quat[:] = sim_data.mocap_quat[i].cpu().numpy()
    mujoco.mj_forward(self.model, self.data)
    with self.viewer.lock():
      self.viewer.cam.lookat[:] = self.data.xpos[self.body]
      scn = self.viewer.user_scn
      scn.ngeom = 0
      z = float(self.data.xpos[self.body][2]) - 0.25
      for name, (pts, rgba) in self.paths.items():
        _segments(
          scn,
          pts,
          z + (0.02 if name == "plan" else 0.0),
          rgba,
          0.02 if name != "route" else 0.012,
        )
      if self.cells is not None:
        pos, rgba, half = self.cells
        size = np.array([half, half, 0.004])
        eye = np.eye(3).flatten()
        for p, c in zip(pos, rgba, strict=True):
          if scn.ngeom >= scn.maxgeom:
            break
          mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, size, p, eye, c
          )
          scn.ngeom += 1
      self.viewer.set_texts(
        (
          mujoco.mjtFontScale.mjFONTSCALE_150.value,
          mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
          self.text,
          "",
        )
      )
    self.viewer.sync()

  def close(self) -> None:
    self.viewer.close()
