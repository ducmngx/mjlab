"""Side-by-side demo video: Nominal MPPI vs ours, same fault at the same place.

  MUJOCO_GL=egl uv run python failbench/paper_figures/fault_video/make_video.py \
      <fault_logs> <fault_results> <clean_dir> <out_dir> [main|baselines|strict ...]

``logs_dir/<arm>/<layout>_s<seed>.npz`` are ``run_mppi.py`` logs of the
household fault experiment (on Hopper:
``/scratch/mnguy21/failbench/mppi/house_faults``); ``results_dir`` holds the
matching run JSONs (``benchmark_results/failbench/house_faults``). A case is
``<layout>_s<seed>:<env>``. ``clean_dir`` is the fault-free runs' folder
(``benchmark_results/failbench/house_clean``: logs and JSONs together).

Writes one clip per case and ``fault_demo_full.mp4``, in three parts (name
some to render only those): ``main``, Nominal MPPI vs ours (``CASES``);
``baselines``, a 2x2 grid with CBF-0.2 and Severity-CBF-0.2
(``BASELINE_CASES``); ``strict``, fault-free runs where the strict margin
stalls the robot (``STRICT_CASES``, at 2x).

A kinematic replay: the scene is rebuilt from the layout and the robots posed
from the logged base pose and joint angles, with the household objects drawn
as figures (``failbench/decor.py``). Both panels share one camera and are
synced at their own fault time; after the fault the clip runs at half speed.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from failbench.env_cfg import failbench_env_cfg
from failbench.faults import LOCK, FaultFamily, faulted_joint
from failbench.house_terrain import HIDDEN_ALPHA, HouseTerrainCfg
from failbench.mppi.cbf import fields_of_levels, severity_levels
from failbench.mppi.costmap import FootprintCfg
from failbench.scripts.play_trial import _add, _line, _yaw
from failbench.sensing import cell_centres
from mjlab.envs import ManagerBasedRlEnv

# Arm, label, colour (the paper charts').
ARMS = (
  ("normal", "Nominal MPPI", (141, 140, 135)),
  ("ours_w1", "Ours: failure-risk critic", (31, 95, 174)),
)
BASELINE_ARMS = (
  ("normal", "Nominal MPPI", (141, 140, 135)),
  ("cbf_0.2", "CBF-0.2", (235, 104, 52)),
  ("cbfsem_0.2", "Severity-CBF-0.2", (27, 175, 122)),
  ("ours_w1", "Ours: failure-risk critic", (31, 95, 174)),
)
STRICT_ARMS = (
  ("cbfsem_0.2", "Severity-CBF-0.2", (27, 175, 122)),
  ("ours_w1", "Ours: failure-risk critic", (31, 95, 174)),
)
# Selection (house_faults JSONs, then the logs): the baseline hits a person
# in the 3 s after the fault, ours touches nothing and doesn't fall, and ours
# reaches the fault's place within ~1 s of the others and is moving at
# >= 0.9 m/s when it fires (so it isn't avoiding the hit by standing still).
CASES = [
  "garden_s3:40",
  "loft_s1:37",
  "crowd_s4:63",
  "apartment_s2:31",
]
BASELINE_CASES = ["crowd_s4:63", "loft_s3:13", "crowd_s1:5", "crowd_s4:45"]
"""As ``CASES``, and at least two of the three others hit a person."""
STRICT_CASES = ["townhouse_s1:39", "loft_s1:30"]
"""Fault-free (house_clean): CBF-0.2 and Severity-CBF-0.2 both stall short of
the goal; ours arrives within 1.3 s of Nominal MPPI."""
PRE_S, POST_S, HOLD_S = 4.0, 3.0, 1.5
"""Seconds shown before the fault and after it (the metrics' window), and
the held last frame."""
WHY_S = (-1.5, 1.5)
"""Window (s, from the failure) over which ours' plans are compared."""
SLOW = 2
"""Slow-motion factor after the fault."""
FAST = 2
"""Speed-up of the fault-free clips."""
FPS = 25
W, H = 1920, 1080
TOP, PAD, BOT = 118, 16, 66
PW, PH = (W - 3 * PAD) // 2, H - TOP - BOT
"""The largest panel (two side by side)."""


def panels(n: int) -> list[tuple[int, int, int, int]]:
  """(x, y, w, h) of each panel: one row of two, or a 2x2 grid."""
  rows = n // 2
  ph = (H - TOP - BOT - (rows - 1) * PAD) // rows
  return [
    (PAD + (i % 2) * (PW + PAD), TOP + (i // 2) * (ph + PAD), PW, ph) for i in range(n)
  ]


FONT = "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf"
BOLD = "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf"
INK, INK2, MUTED = (20, 20, 20), (82, 81, 78), (141, 140, 135)
RED, GREEN = (208, 52, 44), (27, 140, 92)
BG = (250, 250, 248)
GOAL = (25, 150, 70)
TINT = (255, 120, 110)
"""Overlay on a panel once its robot has hit something."""
HIDE = 5
OPT = mujoco.MjvOption()
OPT.geomgroup[HIDE] = 0
JOINT = {"hip": "hip", "thigh": "thigh", "calf": "knee"}
LEG = {"FL": "front-left", "FR": "front-right", "RL": "rear-left", "RR": "rear-right"}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
  return ImageFont.truetype(BOLD if bold else FONT, size)


def fault_words(log, b: int, names: list[str]) -> tuple[str, str]:
  """('rear-right thigh', 'goes limp') from the log's fault."""
  params = log["fault_params"][b]
  j = int(faulted_joint(torch.as_tensor(params)))
  leg, part = names[j].split("_")[:2]
  fam = FaultFamily(int(log["fault_family"][b])).name
  how = "seizes" if params[j, LOCK] > 0.5 else "goes limp" if fam == "WEAK" else fam
  return f"{LEG[leg]} {JOINT[part]}", how


def build(layout: str):
  env_cfg = failbench_env_cfg(terrain=f"house_{layout}")
  env_cfg.scene.num_envs = 1
  gen = env_cfg.scene.terrain.terrain_generator  # type: ignore[union-attr]
  house = gen.sub_terrains.get("house")  # type: ignore[union-attr]
  assert isinstance(house, HouseTerrainCfg)
  house.decor = True
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  m = env.sim.mj_model
  m.vis.global_.offwidth, m.vis.global_.offheight = PW, PH
  # The decorated items' near-invisible collision boxes still cast dithered
  # shadows: move them to a group the renderer leaves out.
  m.geom_group[np.isclose(m.geom_rgba[:, 3], HIDDEN_ALPHA)] = HIDE
  m.vis.quality.shadowsize = 8192
  m.vis.map.shadowscale = 0.3  # Tighter shadow map: no acne on the floor.
  m.vis.map.shadowclip = 3.0
  m.vis.quality.offsamples = 8
  m.vis.headlight.ambient[:] = 0.34
  m.vis.headlight.diffuse[:] = 0.30
  m.vis.headlight.specular[:] = 0.0
  sun = np.array([0.35, 0.25, -0.9])
  for i in range(m.nlight):
    m.light_dir[i] = sun / np.linalg.norm(sun)
    m.light_diffuse[i] = 0.55 if i == 0 else 0.0
    m.light_ambient[i] = 0.0
    m.light_specular[i] = 0.05 if i == 0 else 0.0
    m.light_castshadow[i] = i == 0
  return env, m


class KeepOut:
  """Where a CBF arm's filter won't let the footprint's centre circle go,
  from the planner's own logged maps (one a second): within the circle's
  radius plus the arm's margin (``margin * severity / 25`` for the
  severity-scaled arm) of an obstacle or ledge of each severity, as
  ``failbench/mppi/cbf.py`` computes its barriers."""

  RES = 0.05
  REACH_M = 3.0
  """Cells drawn only this close to the robot (geom budget)."""

  def __init__(self, log, b: int, arm: str):
    self.height = log["map_height"][b].astype(np.float32)
    self.severity = log["map_severity"][b].astype(np.float32)
    every = int(log["map_every"])
    self.map_step = log["cycle_step"][::every][: len(self.height)]
    self.origin = log["cycle_origin_w"][b][::every][: len(self.height)]
    margin = float(arm.split("_")[1])
    self.scaled = arm.startswith("cbfsem")
    self.margin = margin
    self.fp = FootprintCfg()
    self.cache: dict[int, np.ndarray] = {}
    cells = self.height.shape[-1]
    self.xy = cell_centres(cells, self.RES, "cpu").numpy()

  def mask(self, m: int) -> np.ndarray:
    if m not in self.cache:
      h = torch.as_tensor(self.height[m])[None]
      sev = torch.as_tensor(self.severity[m])[None]
      level = severity_levels(h, sev, self.RES, self.fp)
      out = torch.zeros_like(level, dtype=torch.bool)
      for v, dist in fields_of_levels(level, self.RES, self.fp):
        scale = v / 25.0 if self.scaled else 1.0
        out |= dist < self.fp.radius + scale * self.margin
      self.cache[m] = (out & (level == 0))[0].numpy()
    return self.cache[m]

  def draw(self, scene, t: int) -> None:
    ok = np.flatnonzero((self.map_step >= 0) & (self.map_step <= t))
    if not len(ok):
      return
    m = int(ok[-1])
    ox, oy, oz, yaw = self.origin[m]
    iy, ix = np.nonzero(
      self.mask(m) & (np.hypot(*self.xy.transpose(2, 0, 1)) < self.REACH_M)
    )
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = self.xy[iy, ix, 0], self.xy[iy, ix, 1]
    z = oz + self.height[m, iy, ix] + 0.02
    for x, y, zz in zip(ox + c * lx - s * ly, oy + s * lx + c * ly, z, strict=True):
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_BOX,
        (self.RES / 2, self.RES / 2, 0.004),
        (x, y, zz),
        (1.0, 0.45, 0.1, 0.45),
      )


class Run:
  """One arm's robot in one case: its log, fault and outcome."""

  def __init__(self, logs: Path, res: Path, arm: str, name: str, b: int, m):
    log = np.load(logs / arm / f"{name}.npz")
    trial = next(
      t
      for t in json.loads((res / arm / f"{name}.json").read_text())["trials"]
      if t["env"] == b
    )
    self.window = trial.get("window") or {}
    self.reached = bool(trial["reached"])
    self.time_to_goal = trial["time_to_goal_s"]
    self.goal = log["goal_w"][b]
    self.goal_z = 0.0
    self.tick = 0
    """Frames drawn so far (animates the goal)."""
    self.goal_px: tuple[float, float, bool] = (0.0, 0.0, False)
    self.now = 0
    """The step being drawn (for the speed gauge)."""
    self.goal_dist = 0.0
    self.names = [str(n) for n in log["joint_names"]]
    self.dt = float(log["step_dt"])
    j = int(faulted_joint(torch.as_tensor(log["fault_params"][b])))
    # A fault-free run is anchored at the planner's start instead.
    self.faulty = j >= 0
    self.fault = (
      int(log["fault_step"][b])
      if self.faulty
      else round(float(log["start_s"]) / self.dt)
    )
    self.pos = log["dense_base_pos_w"][b]
    self.quat = log["dense_base_quat_w"][b]
    self.q = log["dense_joint_pos"][b]
    self.cmd = log["dense_cmd"][b]
    self.hit_box: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    """(pos, mat, size) of the collision box it hit, drawn red from the hit."""
    self.why: list[str] = []
    """Ours: what its plan did differently after the failure."""
    # Floor under the trunk (its height above the terrain is logged).
    self.ground = self.pos[:, 2] - log["dense_base_height"][b]
    self.path = log["cycle_path_w"][b]
    self.cycle_step = log["cycle_step"]
    self.ran = np.flatnonzero(self.cycle_step >= 0)
    self.what = fault_words(log, b, self.names) if self.faulty else ("", "")
    leg = self.names[j][:2] if self.faulty else "none"
    self.leg_geoms = {
      g
      for g in range(m.ngeom)
      if (
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
      ).startswith(f"robot/{leg}_")
    }
    self.keepout = None
    if arm.startswith("cbf") and not self.faulty:
      self.keepout = KeepOut(log, b, arm)
    hit = self.window.get("first_contact_s")
    self.hit = (
      None if hit is None or math.isnan(hit) else self.fault + round(hit / self.dt)
    )
    base = next(
      i
      for i in range(m.njnt)
      if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE
      and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i).startswith("robot/")
    )
    self.base_adr = m.jnt_qposadr[base]
    self.q_adr = [
      m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{n}")]
      for n in self.names
    ]

  def step(self, rel: int) -> int:
    return int(np.clip(self.fault + rel, 0, len(self.pos) - 1))


def boxes(m, kind: str = "") -> list[int]:
  """The layout's collision boxes (hidden behind their figures), of one
  class if ``kind`` (e.g. ``person``)."""
  out = []
  for g in np.flatnonzero(m.geom_group == HIDE):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if f"_{kind}_" in name or not kind:
      out.append(int(g))
  return out


def planar_gap(p, g: int, m, d) -> float:
  """Planar distance from point ``p`` to box ``g``'s footprint."""
  rel = d.geom_xmat[g].reshape(3, 3).T @ (
    np.array([p[0], p[1], d.geom_xpos[g][2]]) - d.geom_xpos[g]
  )
  q = np.maximum(np.abs(rel[:2]) - m.geom_size[g][:2], 0.0)
  return float(np.hypot(*q))


def pose(run: Run, t: int, m, d) -> None:
  d.qpos[run.base_adr : run.base_adr + 3] = run.pos[t]
  d.qpos[run.base_adr + 3 : run.base_adr + 7] = run.quat[t]
  d.qpos[run.q_adr] = run.q[t]
  mujoco.mj_forward(m, d)


def find_hit_box(run: Run, m, d) -> None:
  """The collision box nearest the robot's collision geoms around its first
  contact (the contact the metrics counted)."""
  if run.hit is None:
    return
  body = [
    g
    for g in range(m.ngeom)
    if (
      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
    ).startswith("robot/")
    and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
  ]
  ft = np.zeros(6)
  best = (np.inf, -1)
  for t in range(max(run.hit - 3, 0), min(run.hit + 4, len(run.pos))):
    pose(run, t, m, d)
    near = [g for g in boxes(m) if planar_gap(run.pos[t], g, m, d) < 0.8]
    for g in near:
      gap = min(mujoco.mj_geomDistance(m, d, a, g, 0.3, ft) for a in body)
      best = min(best, (gap, g))
  if best[1] >= 0:
    g = best[1]
    run.hit_box = (
      d.geom_xpos[g].copy(),
      d.geom_xmat[g].reshape(3, 3).copy(),
      m.geom_size[g].copy(),
    )


def explain(ours: Run, nominal: Run, m, d) -> None:
  """Ours' callout: its plans against Nominal's over ``WHY_S`` around the
  failure, by clearance from people and by commanded speed. Neither planner
  knows of the failure; ours' plans differ because its critic prices where
  an imagined failure would throw the robot."""
  people = boxes(m, "person")
  if not people:
    return

  def plan_stats(run: Run):
    t0 = run.fault + round(WHY_S[0] / run.dt)
    t1 = run.fault + round(WHY_S[1] / run.dt)
    cyc = [
      c
      for c in np.flatnonzero((run.cycle_step >= t0) & (run.cycle_step < t1))
      if not np.isnan(run.path[c, 0, 0])
    ]
    clear = np.mean(
      [
        min(planar_gap(q, g, m, d) for q in run.path[c, :20] for g in people)
        for c in cyc
      ]
    )
    speed = float(np.linalg.norm(run.cmd[t0:t1, :2], axis=-1).mean())
    return clear, speed

  mujoco.mj_forward(m, d)
  co, vo = plan_stats(ours)
  cn, vn = plan_stats(nominal)
  lines = []
  if co - cn >= 0.2:
    lines.append(f"keeps its plan {co:.1f} m clear of people (Nominal {cn:.1f} m)")
  if vo < 0.75 * vn:
    lines.append(f"plans slower, {vo:.1f} m/s (Nominal {vn:.1f} m/s)")
  if lines:
    ours.why = ["Ours anticipates where a failure would throw it:", *lines]


def floor_z(m, d, xy) -> float:
  """Top of whatever is under ``xy`` (terrain or object), by a ray down."""
  groups = np.ones(6, np.uint8)
  groups[HIDE] = 0
  gid = np.zeros(1, np.int32)
  dist = mujoco.mj_ray(
    m,
    d,
    np.array([xy[0], xy[1], 5.0]),
    np.array([0.0, 0.0, -1.0]),
    groups,
    True,
    -1,
    gid,
  )
  return 5.0 - dist if dist >= 0 else 0.0


def project(scene, point, renderer) -> tuple[float, float, bool]:
  """Pixel (x, y) of a world point in the scene's camera, and whether it is
  in front of the camera."""
  cam = scene.camera[0]
  fwd, up = np.array(cam.forward), np.array(cam.up)
  right = np.cross(fwd, up)
  rel = np.asarray(point, float) - np.array(cam.pos)
  zc = float(rel @ fwd)
  if zc <= 1e-6:
    return 0.0, 0.0, False
  n = cam.frustum_near
  hh = (cam.frustum_top - cam.frustum_bottom) / 2
  hw = hh * renderer.width / renderer.height
  u = (float(rel @ right) / zc * n - cam.frustum_center + hw) / (2 * hw)
  v = (cam.frustum_top - float(rel @ up) / zc * n) / (2 * hh)
  return u * renderer.width, v * renderer.height, True


def draw_goal(scene, run: Run, renderer) -> None:
  """The goal, like a game's: a spinning gem bouncing over a pulsing ring, in
  # a faint beam (an arrow at the panel's edge when off-screen). Animated by
  # video time, so it moves alike at 0.5x, 1x and 2x."""
  run.tick += 1
  sec = run.tick / FPS
  g, gz = run.goal, run.goal_z
  _add(
    scene,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    (0.05, 0.6, 0),
    (g[0], g[1], gz + 0.6),
    (0.3, 0.95, 0.45, 0.18),
  )
  ring = 0.28 + 0.07 * math.sin(2 * math.pi * 1.2 * sec)
  _add(
    scene,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    (ring, 0.004, 0),
    (g[0], g[1], gz + 0.015),
    (0.2, 0.85, 0.35, 0.45),
  )
  top = gz + 0.55 + 0.14 * abs(math.sin(math.pi * 1.4 * sec))
  spin = 2 * math.pi * 0.5 * sec
  cz, sz = math.cos(spin), math.sin(spin)
  yaw = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
  a, b = math.radians(45), math.atan(1 / math.sqrt(2))  # Cube on a vertex.
  rx = np.array(
    [[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]]
  )
  ry = np.array(
    [[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]]
  )
  _add(
    scene,
    mujoco.mjtGeom.mjGEOM_BOX,
    (0.09, 0.09, 0.09),
    (g[0], g[1], top),
    (0.15, 0.85, 0.35, 1.0),
    yaw @ ry @ rx,
  )
  run.goal_px = project(scene, (g[0], g[1], top), renderer)


def draw_robot(run: Run, t: int, m, d, scene, renderer, cam, t0: int):
  d.qpos[run.base_adr : run.base_adr + 3] = run.pos[t]
  d.qpos[run.base_adr + 3 : run.base_adr + 7] = run.quat[t]
  d.qpos[run.q_adr] = run.q[t]
  mujoco.mj_forward(m, d)
  renderer.update_scene(d, cam, scene_option=OPT)
  ground = run.ground[t]
  # The trunk's path over the clip: grey while healthy, red once faulted.
  k = np.arange(t0, t + 1, 4)
  for a, e in zip(k[:-1], k[1:], strict=True):
    red = run.faulty and a >= run.fault
    rgba = (0.82, 0.2, 0.17, 0.95) if red else (0.25, 0.25, 0.25, 0.8)
    pa = (*run.pos[a, :2], run.ground[a] + 0.03)
    pe = (*run.pos[e, :2], run.ground[e] + 0.03)
    _line(scene, pa, pe, 0.014, rgba)
  # The plan being followed (green): what the planner commands.
  j = np.searchsorted(run.cycle_step[run.ran], t, side="right") - 1
  if j >= 0 and not np.isnan(run.path[run.ran[j], 0, 0]):
    pts = np.concatenate([run.pos[t : t + 1, :2], run.path[run.ran[j]][:30]], 0)
    pts = np.concatenate([pts, np.full((len(pts), 1), ground + 0.03)], -1)
    for a, e in zip(pts[:-1], pts[1:], strict=True):
      _line(scene, a, e, 0.009, (0.15, 0.7, 0.35, 0.85))
    # A dot every 0.5 s of the plan: closer dots, a slower plan.
    for q in pts[5::5]:
      _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.035, 0, 0), q, (1.0, 1.0, 1.0, 0.95))
  if run.keepout is not None:
    run.keepout.draw(scene, t)
  draw_goal(scene, run, renderer)
  run.goal_dist = float(np.linalg.norm(run.goal[:2] - run.pos[t, :2]))
  # The faulted leg, tinted red once the fault fires.
  if run.faulty and t >= run.fault:
    pulse = 0.75 + 0.25 * math.cos(2 * math.pi * (t - run.fault) * run.dt * 1.5)
    for g in scene.geoms[: scene.ngeom]:
      if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in run.leg_geoms:
        g.rgba[:] = (0.9 * pulse + 0.1, 0.12, 0.1, 1.0)
        g.matid = -1
  if run.hit_box is not None and run.hit is not None and t >= run.hit:  # Hit: red.
    pos, mat, size = run.hit_box
    _add(
      scene, mujoco.mjtGeom.mjGEOM_BOX, size * 1.01, pos, (0.95, 0.1, 0.08, 0.3), mat
    )
  # Where it hit: a red ring that stays.
  if run.hit is not None and t >= run.hit:
    p = run.pos[run.hit]
    age = (t - run.hit) * run.dt
    r = 0.15 + 0.2 * min(age / 0.3, 1.0)
    _add(
      scene,
      mujoco.mjtGeom.mjGEOM_CYLINDER,
      (r, 0.004, 0),
      (p[0], p[1], run.ground[run.hit] + 0.02),
      (0.9, 0.1, 0.08, 0.28),
    )
  return renderer.render()


def chip(dr: ImageDraw.ImageDraw, xy, text: str, fill, fg=(255, 255, 255), size=24):
  f = font(size, bold=True)
  x, y = xy
  w = dr.textlength(text, font=f)
  dr.rounded_rectangle(
    (x, y, x + w + 28, y + size + 18), radius=(size + 18) // 2, fill=fill
  )
  dr.text((x + 14, y + 8), text, font=f, fill=fg)


def goal_arrow(dr, run, x0: int, y0: int, pw: int, ph: int, size: int) -> None:
  """Off-screen goal: a green arrow at the panel's edge pointing to it."""
  gx, gy, front = run.goal_px
  inside = front and 0 <= gx < pw and 0 <= gy < ph
  if inside:
    return
  cx, cy = pw / 2, ph / 2
  dx, dy = (gx - cx, gy - cy) if front else (cx - gx, cy - gy)
  norm = math.hypot(dx, dy) or 1.0
  dx, dy = dx / norm, dy / norm
  m = 34  # Inset from the edge.
  k = min((cx - m) / abs(dx) if dx else 1e9, (cy - m - size) / abs(dy) if dy else 1e9)
  ax, ay = x0 + cx + dx * k, y0 + cy + dy * k
  px, py = -dy, dx
  tip = (ax + dx * 22, ay + dy * 22)
  base = [
    (ax - dx * 8 + px * 15, ay - dy * 8 + py * 15),
    (ax - dx * 8 - px * 15, ay - dy * 8 - py * 15),
  ]
  dr.polygon([tip, *base], fill=GOAL, outline=(255, 255, 255))
  text = f"goal {run.goal_dist:.1f} m"
  f = font(size, bold=True)
  tw = dr.textlength(text, font=f)
  tx = min(max(ax - dx * 30 - tw / 2, x0 + 8), x0 + pw - tw - 8)
  ty = min(max(ay - dy * 30 - size / 2, y0 + 8), y0 + ph - size - 8)
  dr.rounded_rectangle(
    (tx - 8, ty - 4, tx + tw + 8, ty + size + 6), radius=8, fill=(255, 255, 255)
  )
  dr.text((tx, ty), text, font=f, fill=GOAL)


def speed_gauge(dr, run, x1: int, y0: int, size: int) -> None:
  """Top-right of a panel: the plan's commanded speed, a number and a bar."""
  t = getattr(run, "now", 0)
  v = float(np.linalg.norm(run.cmd[t, :2]))
  f = font(size - 2, bold=True)
  text = f"plan {v:.1f} m/s"
  tw = dr.textlength(text, font=f)
  bw = 90 if size > 20 else 64
  x0 = x1 - tw - bw - 44
  dr.rounded_rectangle(
    (x0, y0 + 10, x1 - 10, y0 + 22 + size), radius=8, fill=(255, 255, 255)
  )
  dr.text((x0 + 10, y0 + 14), text, font=f, fill=INK)
  bx, by = x0 + tw + 20, y0 + 16 + size // 2 - 4
  dr.rounded_rectangle((bx, by, bx + bw, by + 9), radius=4, fill=(220, 219, 214))
  dr.rounded_rectangle(
    (bx, by, bx + max(bw * min(v / 1.2, 1.0), 9), by + 9), radius=4, fill=(40, 150, 75)
  )


def why_callout(dr, lines: list[str], x0: int, y0: int, size: int) -> None:
  f, fb = font(size), font(size, bold=True)
  w = max(dr.textlength(t, font=fb if i == 0 else f) for i, t in enumerate(lines))
  h = len(lines) * (size + 8) + 14
  dr.rounded_rectangle((x0, y0, x0 + w + 24, y0 + h), radius=10, fill=(31, 95, 174))
  for i, t in enumerate(lines):
    dr.text(
      (x0 + 12, y0 + 8 + i * (size + 8)),
      t,
      font=fb if i == 0 else f,
      fill=(255, 255, 255),
    )


def canvas_with_panels(imgs, arms, title: str, sub: str, hot=None, runs=None):
  """The page: title, subtitle and each labelled panel. Returns the canvas,
  its drawer, each panel's chip anchor and the chip size."""
  canvas = Image.new("RGB", (W, H), BG)
  dr = ImageDraw.Draw(canvas)
  dr.text((PAD + 4, 18), title, font=font(38, bold=True), fill=INK)
  dr.text((PAD + 4, 68), sub, font=font(24), fill=INK2)
  fs, cs = (28, 24) if len(arms) == 2 else (24, 20)
  anchors = []
  for img, (_, lab, col), (x0, y0, _, ph) in zip(
    imgs, arms, panels(len(arms)), strict=True
  ):
    if hot is not None and hot[len(anchors)]:  # Collided: tint the panel red.
      img = (0.78 * img + 0.22 * np.array(TINT)).astype(np.uint8)
    canvas.paste(Image.fromarray(img), (x0, y0))
    if hot is not None and hot[len(anchors)]:
      dr.rectangle((x0, y0, x0 + img.shape[1] - 1, y0 + ph - 1), outline=RED, width=5)
    dr.rounded_rectangle((x0 + 14, y0 + 14, x0 + 32, y0 + 20 + fs), radius=4, fill=col)
    tw = dr.textlength(lab, font=font(fs, bold=True))
    dr.rounded_rectangle(
      (x0 + 40, y0 + 10, x0 + 60 + tw, y0 + 24 + fs), radius=8, fill=(255, 255, 255)
    )
    dr.text((x0 + 50, y0 + 15), lab, font=font(fs, bold=True), fill=INK)
    if runs is not None:
      goal_arrow(dr, runs[len(anchors)], x0, y0, img.shape[1], ph, cs - 4)
      speed_gauge(dr, runs[len(anchors)], x0 + img.shape[1], y0, cs)
    anchors.append((x0 + 14, y0 + ph - cs - 38))
  return canvas, dr, anchors, cs


def frame(imgs, runs, arms, rel_s: float, speed: str, title: str, sub: str):
  for r in runs:
    r.now = r.step(round(rel_s / r.dt))
  hot = [r.hit is not None and r.now >= r.hit for r in runs]
  canvas, dr, anchors, cs = canvas_with_panels(imgs, arms, title, sub, hot, runs)
  for run, at in zip(runs, anchors, strict=True):
    t = run.step(round(rel_s / run.dt))
    w = run.window
    if rel_s < 0:
      chip(dr, at, "healthy", (90, 90, 88), size=cs)
    elif run.hit is not None and t >= run.hit:
      what = "a person" if w["collided_person"] else "an object"
      chip(dr, at, f"hits {what} at {w['impact_speed']:.1f} m/s", RED, size=cs)
    elif rel_s >= POST_S - 0.05 and run.hit is None:
      text = f"no contact · closest person {w['min_dist_person_m']:.2f} m"
      chip(dr, at, text + (" · falls" if w["fell"] else ""), GREEN, size=cs)
    else:
      chip(dr, at, f"failure: {' '.join(run.what)}", RED, size=cs)
    if run.why:
      why_callout(dr, run.why, at[0], at[1] - 20 - len(run.why) * (cs + 6), cs - 2)
  # Timeline, relative to the fault.
  y = H - BOT + 30
  xa, xb = PAD + 220, W - PAD - 160
  frac = lambda s: xa + (xb - xa) * (s + PRE_S) / (PRE_S + POST_S)  # noqa: E731
  dr.text((PAD + 4, y - 15), f"t = {rel_s:+.1f} s", font=font(26, bold=True), fill=INK)
  dr.line((xa, y, xb, y), fill=(210, 209, 204), width=6)
  dr.line((xa, y, frac(min(rel_s, POST_S)), y), fill=INK2, width=6)
  fx = frac(0)
  dr.line((fx, y - 14, fx, y + 14), fill=RED, width=4)
  dr.text((fx + 8, y - 34), "failure", font=font(20, bold=True), fill=RED)
  dr.text(
    (xb + 18, y - 15),
    speed,
    font=font(26, bold=True),
    fill=RED if speed != "1×" else MUTED,
  )
  return np.asarray(canvas)


def card(lines: list[tuple[str, int, bool, tuple]], seconds: float):
  img = Image.new("RGB", (W, H), BG)
  dr = ImageDraw.Draw(img)
  total = sum(s + 22 for _, s, _, _ in lines)
  y = (H - total) / 2
  for text, size, bold, col in lines:
    f = font(size, bold)
    dr.text(((W - dr.textlength(text, font=f)) / 2, y), text, font=f, fill=col)
    y += size + 22
  return [np.asarray(img)] * round(seconds * FPS)


def clip(
  logs: Path, res: Path, case: str, env_cache: dict, arms=ARMS
) -> list[np.ndarray]:
  name, b = case.split(":")
  b = int(b)
  layout = name.rsplit("_s", 1)[0]
  if layout not in env_cache:
    for e, _ in env_cache.values():
      e.close()
    env_cache.clear()
    env_cache[layout] = build(layout)
  _, m = env_cache[layout]
  d = mujoco.MjData(m)
  runs = [Run(logs, res, arm, name, b, m) for arm, _, _ in arms]
  for r in runs:
    find_hit_box(r, m, d)
  if arms[0][0] == "normal" and arms[-1][0].startswith("ours"):
    explain(runs[-1], runs[0], m, d)
  goal_z = floor_z(m, d, runs[0].goal)
  for r in runs:
    r.goal_z = goal_z
  part, how = runs[0].what
  title = f"The {part} {how} mid-route ({layout.replace('_', ' ')})"
  sub = (
    "Same route, same failure, same place. No planner is told about the failure; "
    "each keeps replanning as usual."
  )
  dt = runs[0].dt
  rels = list(range(-round(PRE_S / dt), 0, 2)) + list(
    range(0, round(POST_S / dt) + 1, 2 // SLOW if SLOW <= 2 else 1)
  )
  if os.environ.get("PREVIEW"):  # A few stills instead of the clip.
    keep = [float(x) for x in os.environ["PREVIEW"].split(",")]
    rels = [round(x / dt) for x in keep]
  renderers = [mujoco.Renderer(m, ph, pw) for _, _, pw, ph in panels(len(arms))]
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  r0 = runs[0]
  # Fixed camera: behind the robots at the fault, high enough to see over
  # people, framing both paths from 1.5 s before the fault to the end.
  heading = math.degrees(_yaw(r0.quat[r0.fault]))
  cam.azimuth = heading + float(os.environ.get("AZ", 0))
  cam.elevation = float(os.environ.get("EL", -55))
  span = np.concatenate(
    [r.pos[r.step(-round(1.5 / dt)) : r.step(round(POST_S / dt)) + 1] for r in runs]
  )
  lo, hi = span.min(0), span.max(0)
  cam.lookat[:] = (lo + hi) / 2 - [0, 0, 0.25]
  cam.distance = max(4.5, 1.5 * float(np.linalg.norm(hi[:2] - lo[:2])) + 2.5)
  cam.distance *= float(os.environ.get("ZOOM", 1.0))
  out = []
  t0s = [r.step(rels[0]) for r in runs]
  for k in rels:
    imgs = []
    for run, rnd, t0 in zip(runs, renderers, t0s, strict=True):
      rnd.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
      imgs.append(draw_robot(run, run.step(k), m, d, rnd.scene, rnd, cam, t0))
    speed = "1×" if k < 0 else f"{1 / SLOW:g}×"
    out.append(frame(imgs, runs, arms, k * dt, speed, title, sub))
  out += [out[-1]] * round(HOLD_S * FPS)
  for rnd in renderers:
    rnd.close()
  return out


def clean_frame(imgs, runs, arms, t_s: float, end_s: float, title: str, sub: str):
  canvas, dr, anchors, cs = canvas_with_panels(imgs, arms, title, sub, runs=runs)
  for run, at in zip(runs, anchors, strict=True):
    if run.reached and t_s >= run.time_to_goal:
      chip(dr, at, f"reached the goal in {run.time_to_goal:.1f} s", GREEN, size=cs)
    elif not run.reached and t_s >= end_s - 0.05:
      left = float(np.linalg.norm(run.goal[:2] - run.pos[-1, :2]))
      text = (
        f"stalled {left:.1f} m short · still there at {len(run.pos) * run.dt:.0f} s"
      )
      chip(dr, at, text, RED, size=cs)
    else:
      chip(dr, at, "driving to the goal", (90, 90, 88), size=cs)
  y = H - BOT + 30
  xa, xb = PAD + 220, W - PAD - 160
  dr.text((PAD + 4, y - 15), f"t = {t_s:.1f} s", font=font(26, bold=True), fill=INK)
  dr.line((xa, y, xb, y), fill=(210, 209, 204), width=6)
  dr.line((xa, y, xa + (xb - xa) * min(t_s / end_s, 1.0), y), fill=INK2, width=6)
  dr.text((xb + 18, y - 15), f"{FAST}×", font=font(26, bold=True), fill=MUTED)
  return np.asarray(canvas)


def clean_clip(clean: Path, case: str, env_cache: dict) -> list[np.ndarray]:
  """A fault-free run: Severity-CBF-0.2 vs ours, start to ours' arrival."""
  name, b = case.split(":")
  b = int(b)
  layout = name.rsplit("_s", 1)[0]
  if layout not in env_cache:
    for e, _ in env_cache.values():
      e.close()
    env_cache.clear()
    env_cache[layout] = build(layout)
  _, m = env_cache[layout]
  d = mujoco.MjData(m)
  runs = [Run(clean, clean, arm, name, b, m) for arm, _, _ in STRICT_ARMS]
  dt = runs[0].dt
  start = runs[0].fault
  # The goal's floor: where the run that arrived ended.
  goal_z = next(r.ground[-1] for r in runs if r.reached)
  for r in runs:
    r.goal_z = goal_z
  end_s = max(r.time_to_goal for r in runs if r.reached) + 1.5
  steps = list(range(start, start + round(end_s / dt) + 1, 2 * FAST))
  if os.environ.get("PREVIEW"):
    steps = [start + round(float(x) / dt) for x in os.environ["PREVIEW"].split(",")]
  renderers = [
    mujoco.Renderer(m, ph, pw, max_geom=30000) for _, _, pw, ph in panels(len(runs))
  ]
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  r0 = runs[0]
  route = r0.goal[:2] - r0.pos[start, :2]
  cam.azimuth = math.degrees(math.atan2(route[1], route[0])) + float(
    os.environ.get("AZ", 0)
  )
  cam.elevation = float(os.environ.get("EL", -60))
  span = np.concatenate([r.pos[start : steps[-1] + 1] for r in runs])
  span = np.concatenate([span, r0.goal[None]])
  lo, hi = span.min(0), span.max(0)
  cam.lookat[:] = (lo + hi) / 2 - [0, 0, 0.25]
  cam.distance = max(4.5, 1.2 * float(np.linalg.norm(hi[:2] - lo[:2])) + 2.5)
  cam.distance *= float(os.environ.get("ZOOM", 1.0))
  title = f"No failure: a strict safety margin can block the route ({layout})"
  sub = (
    "Orange: where Severity-CBF-0.2 won't let the robot go (its footprint plus "
    "0.4 m at people and ledges, 0.2 m at glass)."
  )
  out = []
  for t in steps:
    imgs = []
    for run, rnd in zip(runs, renderers, strict=True):
      rnd.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
      t_ = min(t, len(run.pos) - 1)
      run.now = t_
      imgs.append(draw_robot(run, t_, m, d, rnd.scene, rnd, cam, start))
    out.append(
      clean_frame(imgs, runs, STRICT_ARMS, (t - start) * dt, end_s, title, sub)
    )
  out += [out[-1]] * round(2 * HOLD_S * FPS)
  for rnd in renderers:
    rnd.close()
  return out


def render(out: Path, tag: str, cases: list[str], make) -> list[Path]:
  """Render each case's clip (layouts in order, so each scene is built
  once; with ``REUSE=1``, clips already on disk are kept) and return their
  paths in the given order."""
  paths = [out / f"{tag}{c.replace(':', '_e')}.mp4" for c in cases]
  cache: dict = {}
  for case, path in sorted(zip(cases, paths, strict=True), key=lambda x: x[0]):
    if os.environ.get("REUSE") and path.exists():
      continue
    frames = make(case, cache)
    imageio.mimwrite(path, frames, fps=FPS, quality=8, macro_block_size=1)
    print("done", tag, case, len(frames), flush=True)
  for e, _ in cache.values():
    e.close()
  return paths


def main():
  fault_logs, fault_res, clean, out = (Path(a) for a in sys.argv[1:5])
  only = sys.argv[5:]
  out.mkdir(parents=True, exist_ok=True)

  def part(name):
    return not only or name in only

  reel: list = [
    card(
      [
        ("When a leg fails mid-route", 64, True, INK),
        ("A quadruped in household layouts, planned by MPPI", 32, False, INK2),
        (
          "Failures are injected at the same place for comparison. "
          "No planner is told of the failure.",
          26,
          False,
          MUTED,
        ),
      ],
      4.0,
    )
  ]
  if part("main"):
    reel.append(
      card(
        [
          ("1. Nominal MPPI vs ours", 52, True, INK),
          (
            "Same failure, same place: one hits a person, ours does not",
            30,
            False,
            INK2,
          ),
        ],
        2.5,
      )
    )
    reel += render(
      out, "", CASES, lambda c, cache: clip(fault_logs, fault_res, c, cache, ARMS)
    )
  if part("baselines"):
    reel.append(
      card(
        [
          ("2. Against safety-filter baselines", 52, True, INK),
          (
            "Nominal, CBF-0.2, Severity-CBF-0.2 and ours on the same failure",
            30,
            False,
            INK2,
          ),
        ],
        2.5,
      )
    )
    reel += render(
      out,
      "b_",
      BASELINE_CASES,
      lambda c, cache: clip(fault_logs, fault_res, c, cache, BASELINE_ARMS),
    )
  if part("strict"):
    reel.append(
      card(
        [
          ("3. The price of a strict margin", 52, True, INK),
          ("No failure: Severity-CBF-0.2 vs ours on the same route", 30, False, INK2),
        ],
        2.5,
      )
    )
    reel += render(
      out, "s_", STRICT_CASES, lambda c, cache: clean_clip(clean, c, cache)
    )
  # Streamed: the whole reel doesn't fit in memory.
  path = out / "fault_demo_full.mp4"
  writer = imageio.get_writer(path, fps=FPS, quality=8, macro_block_size=1)
  for piece in reel:
    frames = piece if isinstance(piece, list) else imageio.get_reader(piece).iter_data()
    for f in frames:
      writer.append_data(f)
  writer.close()
  print("saved", path)


if __name__ == "__main__":
  main()
