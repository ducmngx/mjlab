"""Explainer video: how the failure-risk planner weighs plans.

  MUJOCO_GL=egl uv run python failbench/paper_figures/fault_video/explain_video.py \
      <fault_logs> <fault_results> <out.mp4> [case]

Logs and results as ``make_video.py``'s (``<arm>/<layout>_s<seed>.npz`` from
Hopper's ``house_faults``); the case defaults to ``CASE``. A kinematic
replay of ours (``ours_w1``) in one full-screen view:

1. it walks toward the people at full speed;
2. time slows to a stop ``FREEZE_S`` before its real failure and the camera
   pushes in;
3. Plan A, the plan Nominal MPPI chose at that moment in the paired run
   (same route, same place, within a few cm): the eight failures the critic
   imagines, each thrown along the model's predicted path, with the
   critic's damage for each and their expected value;
4. Plan B, ours at that moment, the same;
5. the two side by side;
6. time resumes, the leg really fails (neither planner is told), and the
   paired Nominal run plays as a translucent ghost.

Everything drawn is logged: both runs log the critic's throws and damage
for their chosen plan every cycle (Nominal at weight 0, so they don't steer
it).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from failbench.mppi.risk_critic import RiskCriticCfg, fault_names, imagined_faults
from failbench.paper_figures.fault_video import make_video as mv
from failbench.scripts.play_trial import _add, _line, _yaw

CASE = "garden_s1:16"
FREEZE_S = 0.6
"""Freeze this long before ours' real failure."""
FPS = mv.FPS
W, H = mv.W, mv.H
INK, INK2, MUTED, RED, GREEN = mv.INK, mv.INK2, mv.MUTED, mv.RED, mv.GREEN
PLAN_A = (0.42, 0.44, 0.5)
PLAN_B = (0.12, 0.37, 0.68)
LIMP = (0.95, 0.48, 0.1)
SEIZED = (0.55, 0.3, 0.9)
LEG = mv.LEG
PART = {"hip": "hip", "thigh": "thigh", "calf": "knee"}
PERSON_REACH = 0.25
"""A throw reaches a person when its trunk comes this close to their box
(about the footprint's radius)."""


def ease(x: float) -> float:
  x = min(max(x, 0.0), 1.0)
  return x * x * (3 - 2 * x)


def fault_label(name: str) -> tuple[str, str]:
  joint, kind = name.split(":")
  leg, part = joint.split("_")[:2]
  return f"{LEG[leg]} {PART[part]}", "limp" if kind in ("limp", "free") else "seized"


class Plan:
  """One run's chosen plan at a cycle: its commanded path and the critic's
  imagined throws, shifted to start at ``origin``."""

  def __init__(self, log, b: int, step: int, origin: np.ndarray, start: np.ndarray):
    cs = log["cycle_step"]
    c = int(np.flatnonzero((cs >= 0) & (cs <= step))[-1])
    shift = origin[:2] - start[:2]
    self.path = log["cycle_path_w"][b, c] + shift
    self.throws = log["cycle_risk_thrown_w"][b, c, 0].astype(np.float32) + shift
    self.std = log["cycle_risk_thrown_std"][b, c, 0].astype(np.float32)
    self.damage = log["cycle_risk_damage"][b, c].astype(np.float32)
    self.names = [str(n) for n in log["risk_faults"]]
    cfg = RiskCriticCfg(faults=tuple(self.names))
    params, prior = imagined_faults(cfg, 12)
    odds = dict(zip(fault_names(params), prior.tolist(), strict=True))
    self.prior = np.array([odds[n] for n in self.names])
    self.expected = float((self.prior * self.damage).sum())
    seg = np.linalg.norm(np.diff(self.path[:11], axis=0), axis=1).sum()
    self.speed = float(seg / 1.0)  # The first second of a 0.1 s-step plan.


def person_hits(plan: Plan, m, d) -> list[int | None]:
  """Per throw, the first step that reaches a person (or None)."""
  people = mv.boxes(m, "person")
  out = []
  for f in range(len(plan.throws)):
    hit = None
    for i, q in enumerate(plan.throws[f]):
      if min(mv.planar_gap(q, g, m, d) for g in people) < PERSON_REACH:
        hit = i
        break
    out.append(hit)
  return out


def yaw_mat(th: float) -> np.ndarray:
  c, s = math.cos(th), math.sin(th)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def draw_plan(scene, plan: Plan, z: float, rgb, alpha: float, upto: float = 1.0):
  pts = plan.path[: max(2, round(30 * upto))]
  pts = np.concatenate([pts, np.full((len(pts), 1), z)], -1)
  for a, e in zip(pts[:-1], pts[1:], strict=True):
    _line(scene, a, e, 0.022, (*rgb, alpha))
  for q in pts[5::5]:  # A dot every 0.5 s: closer dots, a slower plan.
    _add(scene, mujoco.mjtGeom.mjGEOM_SPHERE, (0.04, 0, 0), q, (1, 1, 1, alpha))


class Ghosts:
  """Translucent copies of the robot added to a scene: the model's bodies
  posed in a scratch MjData and appended with ``mjv_addGeoms``."""

  def __init__(self, m, run):
    self.m, self.d = m, mujoco.MjData(m)
    self.base, self.q_adr = run.base_adr, run.q_adr
    self.pert = mujoco.MjvPerturb()
    self.legs: dict[str, set[int]] = {}
    for g in range(m.ngeom):
      body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
      if body.startswith("robot/") and body[6:8] in LEG:
        self.legs.setdefault(body[6:8], set()).add(g)

  def add(self, scene, pos, quat, q, rgb, alpha: float, red_leg: str | None = None):
    d = self.d
    d.qpos[self.base : self.base + 3] = pos
    d.qpos[self.base + 3 : self.base + 7] = quat
    d.qpos[self.q_adr] = q
    mujoco.mj_kinematics(self.m, d)
    n0 = scene.ngeom
    mujoco.mjv_addGeoms(
      self.m, d, mv.OPT, self.pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scene
    )
    red = self.legs.get(red_leg or "", set())
    for g in scene.geoms[n0 : scene.ngeom]:
      if g.objtype != mujoco.mjtObj.mjOBJ_GEOM:
        continue
      base = (0.95, 0.12, 0.1) if g.objid in red else rgb
      g.rgba[:] = (
        *[0.35 * c + 0.65 * v for c, v in zip(g.rgba[:3], base, strict=True)],
        alpha,
      )
      g.matid = -1


def yaw_quat(th: float) -> np.ndarray:
  return np.array([math.cos(th / 2), 0.0, 0.0, math.sin(th / 2)])


def draw_throws(scene, plan: Plan, hits, z: float, reveal: list[float], alpha=1.0,
                ghosts: Ghosts | None = None, pose=None):  # fmt: skip
  for f, r in enumerate(reveal):
    if r <= 0:
      continue
    kind = fault_label(plan.names[f])[1]
    rgb = LIMP if kind == "limp" else SEIZED
    n = max(2, round(r * plan.throws.shape[1]))
    pts = plan.throws[f, :n]
    pts = np.concatenate([pts, np.full((len(pts), 1), z)], -1)
    for a, e in zip(pts[:-1], pts[1:], strict=True):
      _line(scene, a, e, 0.016, (*rgb, 0.9 * alpha))
    tip = pts[-1]
    back = pts[max(0, len(pts) - 4)]
    step = tip[:2] - back[:2]
    if ghosts is not None and pose is not None:  # The robot, where it'd be.
      height, heading, q = pose
      th = math.atan2(step[1], step[0]) if np.hypot(*step) > 0.02 else heading
      leg = plan.names[f][:2]
      ghosts.add(scene, (tip[0], tip[1], z - 0.04 + height), yaw_quat(th), q, rgb,
                 0.4 * alpha, red_leg=leg)  # fmt: skip
    h = hits[f]
    if h is not None and n > h:
      p = plan.throws[f, h]
      _add(
        scene,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        (0.12, 0, 0),
        (p[0], p[1], z + 0.15),
        (0.95, 0.12, 0.08, 0.55 * alpha),
      )


def caption(dr, text: str, alpha: float = 1.0) -> None:
  if alpha <= 0 or not text:
    return
  f = mv.font(32)
  lines = text.split("\n")
  tw = max(dr.textlength(t, font=f) for t in lines)
  h = len(lines) * 44 + 24
  x0, y0 = (W - tw) / 2 - 28, H - 70 - h
  fill = (255, 255, 255, round(235 * alpha))
  dr.rounded_rectangle((x0, y0, x0 + tw + 56, y0 + h), radius=16, fill=fill)
  for i, t in enumerate(lines):
    lw = dr.textlength(t, font=f)
    dr.text(
      ((W - lw) / 2, y0 + 14 + 44 * i), t, font=f, fill=(*INK, round(255 * alpha))
    )


def mode_chip(dr, glyph: str, text: str, fill) -> None:
  """Top-left playback state: ``play`` or ``pause`` drawn as a shape."""
  f = mv.font(28, bold=True)
  tw = dr.textlength(text, font=f)
  dr.rounded_rectangle((36, 32, 36 + tw + 76, 32 + 52), radius=26, fill=fill)
  white = (255, 255, 255)
  if glyph == "play":
    dr.polygon([(58, 44), (58, 72), (80, 58)], fill=white)
  else:
    dr.rectangle((58, 45, 65, 71), fill=white)
    dr.rectangle((72, 45, 79, 71), fill=white)
  dr.text((94, 42), text, font=f, fill=white)


def panel(dr, plan: Plan, title: str, rgb, fill: list[float], x0: int = W - 600):
  """The critic's sheet for one plan: per imagined failure, its damage."""
  rows = len(plan.names)
  y0, w = 150, 560
  h = 150 + rows * 50 + 130
  dr.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=18, fill=(255, 255, 255, 238))
  col = tuple(round(255 * v) for v in rgb)
  dr.rounded_rectangle((x0 + 24, y0 + 26, x0 + 40, y0 + 62), radius=4, fill=col)
  dr.text((x0 + 54, y0 + 22), title, font=mv.font(32, bold=True), fill=INK)
  dr.text((x0 + 24, y0 + 78), "If this leg failed now, the damage:", font=mv.font(22),
          fill=INK2)  # fmt: skip
  dr.text((x0 + 24, y0 + 106), "(a person hit at walking speed = 50)",
          font=mv.font(18), fill=MUTED)  # fmt: skip
  shown = 0.0
  for f in range(rows):
    y = y0 + 148 + 50 * f
    where, kind = fault_label(plan.names[f])
    swatch = LIMP if kind == "limp" else SEIZED
    dr.rounded_rectangle((x0 + 24, y + 6, x0 + 38, y + 30), radius=3,
                         fill=tuple(round(255 * v) for v in swatch))  # fmt: skip
    dr.text((x0 + 50, y + 4), f"{where}, {kind}", font=mv.font(22), fill=INK)
    r = fill[f]
    v = float(plan.damage[f]) * r
    shown += float(plan.prior[f]) * v
    bx, bw = x0 + 300, 180
    dr.rounded_rectangle((bx, y + 10, bx + bw, y + 26), radius=6, fill=(228, 227, 222))
    if v > 0.05:
      bar = max(bw * min(v / 50, 1.0), 6)
      hot = RED if v >= 5 else (120, 160, 120)
      dr.rounded_rectangle((bx, y + 10, bx + bar, y + 26), radius=6, fill=hot)
    if r > 0:
      dr.text((bx + bw + 12, y + 4), f"{v:.1f}", font=mv.font(22, bold=True), fill=INK)
  yb = y0 + 148 + 50 * rows + 16
  dr.line((x0 + 24, yb, x0 + w - 24, yb), fill=(220, 219, 214), width=2)
  dr.text((x0 + 24, yb + 16), "Expected damage if a leg fails", font=mv.font(24),
          fill=INK2)  # fmt: skip
  big = mv.font(56, bold=True)
  dr.text((x0 + 24, yb + 46), f"{shown:.1f}", font=big,
          fill=RED if shown >= 5 else GREEN)  # fmt: skip


def compare(dr, a: Plan, b: Plan, progress: float) -> None:
  x0, y0, w, h = W - 600, 150, 560, 470
  dr.rounded_rectangle((x0, y0, x0 + w, y0 + h), radius=18, fill=(255, 255, 255, 238))
  dr.text((x0 + 24, y0 + 22), "Expected damage if a leg fails", font=mv.font(30,
          bold=True), fill=INK)  # fmt: skip
  top = max(a.expected, b.expected)
  # Scores in red (risky) and green (safe), not the plans' path colours.
  for i, (plan, lab, col) in enumerate(((a, "Plan A", RED), (b, "Plan B", GREEN))):
    y = y0 + 100 + 150 * i
    dr.text((x0 + 24, y), f"{lab}  ·  {plan.speed:.1f} m/s", font=mv.font(26,
            bold=True), fill=col)  # fmt: skip
    bw = (w - 180) * plan.expected / top * ease(progress)
    dr.rounded_rectangle((x0 + 24, y + 44, x0 + 24 + max(bw, 8), y + 92), radius=10,
                         fill=col)  # fmt: skip
    dr.text((x0 + 36 + max(bw, 8), y + 50), f"{plan.expected * ease(progress):.1f}",
            font=mv.font(34, bold=True), fill=INK)  # fmt: skip
  ratio = a.expected / max(b.expected, 1e-6)
  dr.text((x0 + 24, y0 + h - 64), f"Plan B: {ratio:.0f}x less risk,", font=mv.font(22),
          fill=INK2)  # fmt: skip
  dr.text((x0 + 24, y0 + h - 36), "for a short detour and a slower walk.",
          font=mv.font(22), fill=INK2)  # fmt: skip


def main():
  logs, res, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
  case = sys.argv[4] if len(sys.argv) > 4 else CASE
  name, b = case.split(":")
  b = int(b)
  layout = name.rsplit("_s", 1)[0]
  env, m = mv.build(layout)
  m.vis.global_.offwidth, m.vis.global_.offheight = W, H
  d = mujoco.MjData(m)
  ours = mv.Run(logs, res, "ours_w1", name, b, m)
  nom = mv.Run(logs, res, "normal", name, b, m)
  mv.find_hit_box(nom, m, d)
  goal_z = mv.floor_z(m, d, ours.goal)
  ours.goal_z = nom.goal_z = goal_z
  log_b = np.load(logs / "ours_w1" / f"{name}.npz")
  log_a = np.load(logs / "normal" / f"{name}.npz")
  dt = ours.dt
  back = float(sys.argv[5]) if len(sys.argv) > 5 else FREEZE_S
  freeze = ours.fault - round(back / dt)
  freeze_a = nom.fault - round(back / dt)
  here = ours.pos[freeze]
  plan_b = Plan(log_b, b, freeze, here, here)
  plan_a = Plan(log_a, b, freeze_a, here, nom.pos[freeze_a])
  mujoco.mj_forward(m, d)
  hits_a, hits_b = person_hits(plan_a, m, d), person_hits(plan_b, m, d)
  z = ours.ground[freeze] + 0.04
  print(f"plan A {plan_a.expected:.1f} ({plan_a.speed:.2f} m/s), "
        f"plan B {plan_b.expected:.1f} ({plan_b.speed:.2f} m/s)")  # fmt: skip

  heading = _yaw(ours.quat[freeze])
  fwd = np.array([math.cos(heading), math.sin(heading), 0.0])
  right = np.array([math.sin(heading), -math.cos(heading), 0.0])
  # Camera: wide (following the robot) and close (on the frozen moment).
  wide_dist, wide_el = 8.5, -50.0
  close_look = here + 0.9 * fwd + 1.4 * right - [0, 0, 0.25]
  close_dist, close_el = 5.8, -62.0

  renderer = mujoco.Renderer(m, H, W, max_geom=20000)
  ghosts = Ghosts(m, ours)
  stance = (ours.pos[freeze, 2] - ours.ground[freeze], heading, ours.q[freeze])
  nom_leg = next((k for k, v in ghosts.legs.items() if v & nom.leg_geoms), None)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE
  cam.azimuth = math.degrees(heading)
  t_start = freeze - round(4.0 / dt)
  smooth = ours.pos[t_start].copy()

  def set_cam(t: int, zoom: float):
    nonlocal smooth
    smooth += 0.08 * (ours.pos[t] + 1.2 * fwd - smooth)
    look_w = smooth - [0, 0, 0.2]
    k = ease(zoom)
    cam.lookat[:] = (1 - k) * look_w + k * close_look
    cam.distance = (1 - k) * wide_dist + k * close_dist
    cam.elevation = (1 - k) * wide_el + k * close_el

  def render(t: int, extras=None, ghost_t: int | None = None, trail=True):
    mv.pose(ours, t, m, d)
    renderer.update_scene(d, cam, scene_option=mv.OPT)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    scene = renderer.scene
    if trail:
      k = np.arange(t_start, t + 1, 4)
      for a, e in zip(k[:-1], k[1:], strict=True):
        red = a >= ours.fault
        rgba = (0.82, 0.2, 0.17, 0.95) if red else (0.25, 0.25, 0.25, 0.8)
        _line(scene, (*ours.pos[a, :2], ours.ground[a] + 0.03),
              (*ours.pos[e, :2], ours.ground[e] + 0.03), 0.014, rgba)  # fmt: skip
    mv.draw_goal(scene, ours, renderer)
    if t >= ours.fault:
      for g in scene.geoms[: scene.ngeom]:
        if g.objtype == mujoco.mjtObj.mjOBJ_GEOM and g.objid in ours.leg_geoms:
          g.rgba[:] = (0.95, 0.12, 0.1, 1.0)
          g.matid = -1
    if extras:
      extras(scene)
    if ghost_t is not None:  # Nominal's trail and the box it hit.
      if nom.hit_box is not None and nom.hit is not None and ghost_t >= nom.hit:
        pos, mat, size = nom.hit_box
        _add(
          scene,
          mujoco.mjtGeom.mjGEOM_BOX,
          size * 1.01,
          pos,
          (0.95, 0.1, 0.08, 0.3),
          mat,
        )
      k = np.arange(nom.fault - round(1.0 / dt), ghost_t + 1, 4)
      for a, e in zip(k[:-1], k[1:], strict=True):
        _line(
          scene,
          (*nom.pos[a, :2], nom.ground[a] + 0.03),
          (*nom.pos[e, :2], nom.ground[e] + 0.03),
          0.014,
          (*PLAN_A, 0.9),
        )
      # Nominal's robot (Plan A), a translucent ghost.
      ghosts.add(scene, nom.pos[ghost_t], nom.quat[ghost_t], nom.q[ghost_t],
                 (0.9, 0.4, 0.35), 0.5, red_leg=nom_leg)  # fmt: skip
    img = renderer.render().astype(np.float32)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).convert("RGBA")

  def finish(img: Image.Image, draw_fn) -> np.ndarray:
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    draw_fn(dr)
    return np.asarray(Image.alpha_composite(img, over).convert("RGB"))

  if os.environ.get("PREVIEW"):  # Stills of both plans, fully imagined.
    set_cam(freeze, 1.0)
    for tag, plan, hits, rgb in (
      ("A", plan_a, hits_a, PLAN_A),
      ("B", plan_b, hits_b, PLAN_B),
    ):

      def extras(scene, plan=plan, hits=hits, rgb=rgb):
        draw_plan(scene, plan, z + 0.005, rgb, 0.95)
        draw_throws(scene, plan, hits, z + 0.01, [1.0] * len(plan.names),
                    ghosts=ghosts, pose=stance)  # fmt: skip

      img = render(freeze, extras)
      frame = finish(img, lambda dr, plan=plan, rgb=rgb, tag=tag: panel(
        dr, plan, f"Plan {tag}", rgb, [1.0] * len(plan.names)))  # fmt: skip
      imageio.imwrite(out.with_name(f"{out.stem}_{tag}.png"), frame)
    env.close()
    return
  writer = imageio.get_writer(out, fps=FPS, quality=8, macro_block_size=1)

  def emit(frame: np.ndarray, n: int = 1):
    for _ in range(n):
      writer.append_data(frame)

  def card(lines, seconds):
    for f in mv.card(lines, seconds):
      emit(f)

  # 0. Title.
  card(
    [
      ("How does the robot plan for a failure", 60, True, INK),
      ("before it happens?", 60, True, INK),
      ("A failure-impact mitigation approach to navigation", 30, False, INK2),
    ],
    3.5,
  )

  # 1. Walking, then time slows to a stop and the camera pushes in.
  rates = [2.0] * 70 + [2.0 * (1 - i / 30) ** 1.5 for i in range(30)]
  steps = np.cumsum([0.0, *rates])
  steps = freeze - (steps[-1] - steps)
  for i, s in enumerate(steps):
    t = int(round(s))
    zoom = (i - 50) / (len(steps) - 50)
    set_cam(t, zoom)
    speed = rates[min(i, len(rates) - 1)] / 2
    text = (f"A legged robot walks to its goal through the {layout.replace('_', ' ')},\n"
            "with people nearby. Its legs are healthy.")  # fmt: skip
    img = render(t)
    emit(finish(img, lambda dr, s=speed, tx=text: (
      mode_chip(dr, "play", "1×" if s > 0.95 else f"slowing… {s:.1f}×", (60, 60, 58)),
      caption(dr, tx))))  # fmt: skip

  # 2. Frozen: the question.
  set_cam(freeze, 1.0)
  base = render(freeze)
  q = ("Every 0.1 s, before it commits to a plan, the planner asks:\n"
       "what if one of my legs failed right now?")  # fmt: skip
  for i in range(round(3.0 * FPS)):
    emit(finish(base, lambda dr, i=i: (
      mode_chip(dr, "pause", "paused · imagining", (31, 95, 174)),
      caption(dr, q, min(1.0, i / 8)))))  # fmt: skip

  # 3/4. Each plan's imagined failures, thrown one after another.
  def imagine(plan, hits, rgb, title, text, keep=None):
    n = len(plan.names)
    grow, stagger, hold = 36, 7, round(3.5 * FPS)
    total = grow + stagger * (n - 1) + hold
    for i in range(total):
      reveal = [ease((i - stagger * f) / grow) for f in range(n)]
      path_in = ease(i / 12)

      def extras(scene, reveal=reveal, path_in=path_in):
        if keep is not None:
          draw_plan(scene, keep, z, PLAN_A, 0.35)
        draw_plan(scene, plan, z + 0.005, rgb, 0.95, path_in)
        draw_throws(scene, plan, hits, z + 0.01, reveal, ghosts=ghosts, pose=stance)

      img = render(freeze, extras)
      emit(finish(img, lambda dr, reveal=reveal: (
        mode_chip(dr, "pause", "paused · imagining", (31, 95, 174)),
        panel(dr, plan, title, rgb, reveal),
        caption(dr, text))))  # fmt: skip

  a_hits = sum(h is not None for h in hits_a)
  imagine(plan_a, hits_a, PLAN_A, f"Plan A · {plan_a.speed:.1f} m/s",
          "Plan A heads straight on. A learned model predicts where each failure\n"
          f"would throw the robot: {a_hits} of 8 throws reach a person.")  # fmt: skip
  b_hits = sum(h is not None for h in hits_b)
  imagine(plan_b, hits_b, PLAN_B, f"Plan B · {plan_b.speed:.1f} m/s",
          "Plan B swings wider and walks slower.\n"
          + (
            "Now none of the 8 throws reach a person."
            if b_hits == 0
            else f"Now only {b_hits} of 8 throws reach a person, and slowly."
          ), keep=plan_a)  # fmt: skip

  # 5. Side by side.
  def both(scene):
    draw_plan(scene, plan_a, z, PLAN_A, 0.9)
    draw_plan(scene, plan_b, z + 0.005, PLAN_B, 0.95)

  base = render(freeze, both)
  pick = (
    "MPPI weighs hundreds of candidate plans like these, adding this risk to its\n"
    "usual costs (reaching the goal, following the route). It picks Plan B."
  )
  for i in range(round(5.0 * FPS)):
    emit(finish(base, lambda dr, i=i: (
      mode_chip(dr, "pause", "paused · choosing", (31, 95, 174)),
      compare(dr, plan_a, plan_b, i / 20),
      caption(dr, pick))))  # fmt: skip

  # 6. Resume; the failure really happens; Nominal (Plan A) as a ghost.
  post = round(3.0 / dt)
  frames = [freeze + k for k in range(0, ours.fault - freeze, 2)]
  frames += [ours.fault + k for k in range(0, post + 1)]
  img, text = render(freeze), ""
  for i, t in enumerate(frames):
    zoom = 1.0 - 0.35 * ease(i / 40)
    set_cam(t, zoom)
    rel = t - ours.fault
    ghost = nom.fault + rel
    img = render(t, ghost_t=ghost)
    failed = rel >= 0
    hit = nom.hit is not None and ghost >= nom.hit
    if not failed:
      text = "It walks on with Plan B. Then…"
    elif not hit:
      text = (
        f"Then the {ours.what[0]} really {ours.what[1]}. Neither planner is told.\n"
        "Ghost: the standard planner, which took Plan A, in the same failure."
      )
    else:
      v = nom.window["impact_speed"]
      dp = ours.window["min_dist_person_m"]
      text = (
        f"Plan A (ghost) hits a person at {v:.1f} m/s.\n"
        f"Ours, on Plan B, stays {dp:.2f} m clear."
      )
    speed = "0.5×" if failed else "1×"
    emit(finish(img, lambda dr, tx=text, sp=speed, f=failed: (
      mode_chip(dr, "play", sp, RED if f else (60, 60, 58)),
      caption(dr, tx))))  # fmt: skip
  emit(finish(img, lambda dr: caption(dr, text)), round(1.5 * FPS))

  # 7. The idea.
  card([("Anticipate, don't react", 60, True, INK),
        ("The planner never knows when or which leg will fail.", 30, False, INK2),
        ("It prices the impact of each failure,", 30, False, INK2),
        ("and prefers plans where that impact can be mitigated.", 30, False, INK2)],
       4.5)  # fmt: skip
  writer.close()
  renderer.close()
  env.close()
  print("saved", out)


if __name__ == "__main__":
  main()
