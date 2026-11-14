"""Look-only dressing for the household layouts' objects.

Each object in ``failbench/house.py`` is a box, and the box is all that is
checked: contacts, the planners' scans and the critics' severities. With
``HouseTerrainCfg(decor=True)`` the box is hidden (still there, still
group 0) and drawn instead as a stylised figure made of primitives chosen by
the item's label: a person with a head, arms and legs, a sofa with a seat,
back and arms, a glass table with a clear top on thin legs. Every part lies
inside the box (``part_bounds``), so what is drawn is what can be hit, and
none collides or is scanned: contype and conaffinity 0, geom group 2.

Parts are laid out in the box's own frame: ``u`` along its longer side,
``v`` across, ``z`` up from the ground it stands on. Colours follow the
paper's classes where they carry meaning: people dress in the person
orange, glass is the glass blue; furniture is wood and fabric.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from failbench.house import Item

Rgba = tuple[float, float, float, float]

GROUP = 2
GLASS: Rgba = (0.55, 0.75, 0.95, 0.3)
GLASS_EDGE: Rgba = (0.16, 0.47, 0.84, 0.9)
GLASS_SHELF: Rgba = (0.50, 0.70, 0.92, 0.45)
METAL: Rgba = (0.72, 0.73, 0.75, 1.0)
DARK: Rgba = (0.12, 0.12, 0.13, 1.0)
WOOD: Rgba = (0.66, 0.49, 0.33, 1.0)
DARK_WOOD: Rgba = (0.42, 0.29, 0.19, 1.0)
CREAM: Rgba = (0.95, 0.92, 0.84, 1.0)
LEAF: Rgba = (0.33, 0.56, 0.27, 1.0)
LEAF_DARK: Rgba = (0.22, 0.43, 0.20, 1.0)
TERRACOTTA: Rgba = (0.74, 0.42, 0.27, 1.0)
CARDBOARD: Rgba = (0.77, 0.61, 0.41, 1.0)
TAPE: Rgba = (0.88, 0.80, 0.62, 1.0)
FABRICS: tuple[Rgba, ...] = (
  (0.36, 0.45, 0.56, 1.0),
  (0.56, 0.43, 0.38, 1.0),
  (0.44, 0.52, 0.42, 1.0),
  (0.62, 0.60, 0.55, 1.0),
)
SHIRTS: tuple[Rgba, ...] = (  # Shades of the person orange (#eb6834).
  (0.92, 0.41, 0.20, 1.0),
  (0.85, 0.34, 0.16, 1.0),
  (0.94, 0.54, 0.36, 1.0),
  (0.79, 0.31, 0.12, 1.0),
)
TROUSERS: tuple[Rgba, ...] = (
  (0.20, 0.26, 0.38, 1.0),
  (0.28, 0.28, 0.30, 1.0),
  (0.45, 0.40, 0.33, 1.0),
)
SKINS: tuple[Rgba, ...] = (
  (0.96, 0.80, 0.69, 1.0),
  (0.87, 0.67, 0.52, 1.0),
  (0.66, 0.46, 0.33, 1.0),
  (0.45, 0.31, 0.22, 1.0),
)
HAIRS: tuple[Rgba, ...] = (
  (0.15, 0.11, 0.08, 1.0),
  (0.35, 0.22, 0.12, 1.0),
  (0.80, 0.66, 0.40, 1.0),
  (0.55, 0.55, 0.55, 1.0),
)


@dataclass(frozen=True)
class Part:
  """A primitive in layout coordinates: ``box``/``ellipsoid`` at ``a`` with
  half sizes ``b``; ``sphere`` at ``a`` of radius ``r``; ``capsule``/
  ``cylinder`` from ``a`` to ``b`` of radius ``r``."""

  kind: str
  a: tuple[float, float, float]
  b: tuple[float, float, float]
  r: float
  rgba: Rgba


def part_bounds(p: Part) -> tuple[np.ndarray, np.ndarray]:
  """Axis-aligned bounds (lo, hi) of ``p`` in layout coordinates."""
  a, b = np.array(p.a), np.array(p.b)
  if p.kind in ("box", "ellipsoid"):
    return a - b, a + b
  if p.kind == "sphere":
    return a - p.r, a + p.r
  if p.kind == "capsule":
    return np.minimum(a, b) - p.r, np.maximum(a, b) + p.r
  # A cylinder: round its axis, flat ends (axis-aligned here).
  axis = np.abs(b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
  pad = p.r * np.sqrt(np.clip(1.0 - axis**2, 0.0, 1.0))
  return np.minimum(a, b) - pad, np.maximum(a, b) + pad


class _Frame:
  """Builds parts in the box's ``(u, v, z)`` frame."""

  def __init__(self, it: Item):
    self.along_x = (it.x1 - it.x0) >= (it.y1 - it.y0)
    self.cx, self.cy = (it.x0 + it.x1) / 2, (it.y0 + it.y1) / 2
    self.z0 = it.z0
    self.L = max(it.x1 - it.x0, it.y1 - it.y0)
    self.W = min(it.x1 - it.x0, it.y1 - it.y0)
    self.H = it.z1 - it.z0
    self.parts: list[Part] = []

  def _pt(self, u: float, v: float, z: float) -> tuple[float, float, float]:
    x, y = (u, v) if self.along_x else (v, u)
    return (self.cx + x, self.cy + y, self.z0 + z)

  def _half(self, hu: float, hv: float, hz: float) -> tuple[float, float, float]:
    return (hu, hv, hz) if self.along_x else (hv, hu, hz)

  def box(self, u, v, z, hu, hv, hz, rgba, kind="box"):
    self.parts.append(Part(kind, self._pt(u, v, z), self._half(hu, hv, hz), 0.0, rgba))

  def slab(self, u0, u1, v0, v1, z0, z1, rgba, kind="box"):
    """A box spanning ``[u0, u1] x [v0, v1] x [z0, z1]``."""
    self.box(
      (u0 + u1) / 2, (v0 + v1) / 2, (z0 + z1) / 2,
      (u1 - u0) / 2, (v1 - v0) / 2, (z1 - z0) / 2, rgba, kind,
    )  # fmt: skip

  def sphere(self, u, v, z, r, rgba):
    self.parts.append(Part("sphere", self._pt(u, v, z), (0.0, 0.0, 0.0), r, rgba))

  def rod(self, p, q, r, rgba, kind="capsule"):
    self.parts.append(Part(kind, self._pt(*p), self._pt(*q), r, rgba))


def _pick(options, rng):
  return options[int(rng.integers(len(options)))]


def _human(f: _Frame, rng, child: bool = False) -> None:
  h = f.H
  head_r = (0.085 if child else 0.065) * h
  sw = min((0.30 if child else 0.26) * h, 0.95 * f.L)  # Shoulder width.
  depth = min((0.13 if child else 0.11) * h, 0.9 * f.W)
  skin, shirt = _pick(SKINS, rng), _pick(SHIRTS, rng)
  legs, hair = _pick(TROUSERS, rng), _pick(HAIRS, rng)
  hip = (0.42 if child else 0.48) * h
  shoulder = (0.74 if child else 0.80) * h
  leg_r = min(0.045 * h, sw * 0.2, depth / 2)
  arm_r = min(0.032 * h, sw * 0.12, depth / 2)
  # Legs and shoes.
  for s in (-1, 1):
    lu = s * sw * 0.2
    f.rod((lu, 0, leg_r), (lu, 0, hip), leg_r, legs)
    f.box(lu, 0, 0.025, leg_r * 1.1, min(leg_r * 1.6, f.W / 2), 0.025, DARK)
  # Torso, slightly narrower at the hips.
  f.slab(-sw / 2 + arm_r * 2, sw / 2 - arm_r * 2, -depth / 2, depth / 2,
         hip - 0.02 * h, shoulder, shirt)  # fmt: skip
  f.rod((-sw / 2 + arm_r * 2, 0, shoulder - arm_r),
        (sw / 2 - arm_r * 2, 0, shoulder - arm_r), arm_r * 1.3, shirt)  # fmt: skip
  # Arms hanging at the sides, sleeves then hands.
  for s in (-1, 1):
    au = s * (sw / 2 - arm_r)
    elbow = shoulder - 0.17 * h
    f.rod((au, 0, shoulder - arm_r), (au, 0, elbow), arm_r, shirt)
    f.rod((au, 0, elbow), (au, 0, hip - 0.05 * h), arm_r * 0.85, skin)
  # Neck, head and hair.
  neck_top = shoulder + 0.04 * h
  f.rod((0, 0, shoulder), (0, 0, neck_top), min(0.025 * h, depth / 2), skin, "cylinder")
  hz = min(neck_top + head_r, h - head_r * 1.05)
  f.sphere(0, 0, hz, head_r, skin)
  f.box(0, 0, hz + head_r * 0.35, head_r * 1.02, head_r * 1.02, head_r * 0.68, hair,
        "ellipsoid")  # fmt: skip


def _table(f: _Frame, top: Rgba, legs: Rgba, thick: float = 0.04) -> None:
  r = min(0.025, f.W / 8)
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, f.H - thick, f.H, top)
  iu, iv = f.L / 2 - r - 0.03, f.W / 2 - r - 0.03
  for su in (-1, 1):
    for sv in (-1, 1):
      f.rod((su * iu, sv * iv, 0), (su * iu, sv * iv, f.H - thick), r, legs, "cylinder")


def _desk(f: _Frame) -> None:
  _table(f, WOOD, DARK_WOOD)
  f.slab(f.L / 2 - 0.4 * f.L, f.L / 2 - 0.02, -f.W / 2 + 0.02, f.W / 2 - 0.02,
         0.05, f.H - 0.04, DARK_WOOD)  # fmt: skip


def _counter(f: _Frame, rgba: Rgba) -> None:
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, f.H - 0.05, f.H, CREAM)
  f.slab(-f.L / 2 + 0.03, f.L / 2 - 0.03, -f.W / 2 + 0.06, f.W / 2,
         0.0, f.H - 0.05, rgba)  # fmt: skip


def _sofa(f: _Frame, rng) -> None:
  fabric = _pick(FABRICS, rng)
  light = (*(min(1.0, c + 0.08) for c in fabric[:3]), 1.0)
  h, arm = f.H, min(0.16, f.L * 0.12)
  back = min(0.2, f.W * 0.25)
  seat = 0.45 * h
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0.06, seat * 0.75, fabric)
  for su in (-1, 1):  # Feet.
    for sv in (-1, 1):
      f.box(su * (f.L / 2 - 0.05), sv * (f.W / 2 - 0.05), 0.03, 0.03, 0.03, 0.03,
            DARK_WOOD)  # fmt: skip
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, -f.W / 2 + back, seat * 0.75, h, fabric)
  for s in (-1, 1):
    u0 = s * f.L / 2
    f.slab(min(u0, u0 - s * arm), max(u0, u0 - s * arm), -f.W / 2, f.W / 2,
           seat * 0.75, 0.68 * h, fabric)  # fmt: skip
  n = max(1, round((f.L - 2 * arm) / 0.65))
  w = (f.L - 2 * arm) / n
  for k in range(n):
    u = -f.L / 2 + arm + (k + 0.5) * w
    f.box(u, (back) / 2, (seat * 0.75 + seat) / 2, w / 2 - 0.01,
          (f.W - back) / 2 - 0.01, (seat - seat * 0.75) / 2, light)  # fmt: skip


def _bed(f: _Frame, rng) -> None:
  h = f.H
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0.05, 0.3 * h, DARK_WOOD)
  f.slab(-f.L / 2 + 0.06, f.L / 2, -f.W / 2 + 0.02, f.W / 2 - 0.02, 0.3 * h, 0.55 * h,
         CREAM)  # fmt: skip
  f.slab(-f.L / 2, -f.L / 2 + 0.06, -f.W / 2, f.W / 2, 0.05, h, DARK_WOOD)
  for s in (-1, 1):
    f.box(-f.L / 2 + 0.25, s * f.W / 4, 0.62 * h, 0.15, f.W / 4 - 0.04,
          0.07 * h, (1.0, 1.0, 1.0, 1.0), "ellipsoid")  # fmt: skip
  f.slab(-f.L / 2 + 0.5, f.L / 2, -f.W / 2 + 0.01, f.W / 2 - 0.01, 0.3 * h, 0.6 * h,
         _pick(FABRICS, rng))  # fmt: skip


def _chairs(f: _Frame, rng) -> None:
  n = max(1, round(f.L / 0.55))
  w = f.L / n
  color = _pick((WOOD, DARK_WOOD), rng)
  seat = min(0.46, 0.55 * f.H)
  for k in range(n):
    c = -f.L / 2 + (k + 0.5) * w
    hu, hv = min(0.22, w / 2 - 0.02), min(0.22, f.W / 2 - 0.02)
    f.slab(c - hu, c + hu, -hv, hv, seat - 0.04, seat, color)
    for su in (-1, 1):
      for sv in (-1, 1):
        f.rod((c + su * (hu - 0.02), sv * (hv - 0.02), 0),
              (c + su * (hu - 0.02), sv * (hv - 0.02), seat - 0.04), 0.015,
              color, "cylinder")  # fmt: skip
    f.slab(c - hu, c + hu, -hv, -hv + 0.04, seat, f.H, color)


def _shelves(f: _Frame, rng, books: bool = False) -> None:
  t = 0.025
  body = _pick((WOOD, DARK_WOOD, CREAM), rng)
  # Sides, top, bottom and back.
  for s in (-1, 1):
    u0 = s * f.L / 2
    f.slab(min(u0, u0 - s * t), max(u0, u0 - s * t), -f.W / 2, f.W / 2, 0, f.H, body)
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, f.H - t, f.H, body)
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0, 0.06, body)
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, -f.W / 2 + t, 0, f.H, body)
  n = max(1, round(f.H / 0.38))
  for k in range(1, n):
    z = k * f.H / n
    f.slab(-f.L / 2 + t, f.L / 2 - t, -f.W / 2 + t, f.W / 2, z - t / 2, z + t / 2,
           body)  # fmt: skip
  if books:
    for k in range(n):
      z0 = 0.06 + k * f.H / n
      u = -f.L / 2 + t + 0.02
      while u < f.L / 2 - t - 0.06:
        w = float(rng.uniform(0.025, 0.05))
        hb = float(rng.uniform(0.55, 0.8)) * (f.H / n - 0.04)
        f.slab(u, u + w, -f.W / 2 + t + 0.01, f.W / 2 - 0.03, z0, z0 + hb,
               _pick(FABRICS + SHIRTS[:1] + (CREAM,), rng))  # fmt: skip
        u += w + 0.004


def _vitrine(f: _Frame, rng) -> None:
  """A glass cabinet: a thin metal frame, clear panes, glass shelves with a
  few things on them."""
  r = min(0.012, f.W / 8, f.L / 8)
  u, v = f.L / 2 - r, f.W / 2 - r
  for su in (-1, 1):  # Uprights.
    for sv in (-1, 1):
      f.rod((su * u, sv * v, 0), (su * u, sv * v, f.H), r, METAL, "cylinder")
  for z in (r, f.H - r):  # Top and bottom rims.
    for sv in (-1, 1):
      f.rod((-u, sv * v, z), (u, sv * v, z), r, METAL, "cylinder")
    for su in (-1, 1):
      f.rod((su * u, -v, z), (su * u, v, z), r, METAL, "cylinder")
  f.slab(-u, u, -v, v, 0, f.H, GLASS)
  n = max(1, round(f.H / 0.4))
  for k in range(1, n + 1):
    z = k * f.H / n - (0.0 if k < n else 2 * r)
    if k < n:
      f.slab(-u, u, -v, v, z - 0.006, z + 0.006, GLASS_SHELF)
    z0 = (k - 1) * f.H / n + (0.006 if k > 1 else 2 * r)
    room = z - z0 - 0.01
    for c in np.linspace(-u * 0.6, u * 0.6, max(1, round(f.L / 0.3))):
      hz = room * float(rng.uniform(0.3, 0.6)) / 2
      rr = min(0.05, v * 0.6)
      f.box(float(c), 0, z0 + hz, rr, rr, hz,
            _pick(SHIRTS + FABRICS + (CREAM,), rng), "ellipsoid")  # fmt: skip


def _cabinet(f: _Frame, rng) -> None:
  body = _pick((WOOD, DARK_WOOD, CREAM), rng)
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0.05, f.H, body)
  n = max(1, round(f.L / 0.5))
  for k in range(n):  # Door fronts and handles.
    u0 = -f.L / 2 + k * f.L / n
    f.slab(u0 + 0.01, u0 + f.L / n - 0.01, f.W / 2 - 0.01, f.W / 2, 0.07, f.H - 0.03,
           (*(min(1.0, c + 0.06) for c in body[:3]), 1.0))  # fmt: skip
    f.box(u0 + f.L / n / 2, f.W / 2 - 0.005, f.H * 0.75, 0.04, 0.005, 0.008, METAL)


def _plant(f: _Frame, rng, pot: Rgba = TERRACOTTA) -> None:
  r = min(f.L, f.W) / 2
  pot_h = min(0.35 * f.H, 0.4)
  f.rod((0, 0, 0), (0, 0, pot_h), r * 0.6, pot, "cylinder")
  f.rod((0, 0, pot_h - 0.02), (0, 0, pot_h), r * 0.66, pot, "cylinder")
  top = f.H - pot_h
  for k in range(4):
    rr = r * float(rng.uniform(0.55, 0.85))
    hz = top * float(rng.uniform(0.25, 0.35))
    ang = k * np.pi / 2 + float(rng.uniform(-0.4, 0.4))
    off = (r - rr) * 0.9
    zc = min(pot_h + hz + k * top * 0.12, f.H - hz)
    f.box(off * np.cos(ang), off * np.sin(ang), zc, rr, rr, hz,
          LEAF if k % 2 else LEAF_DARK, "ellipsoid")  # fmt: skip


def _vase(f: _Frame, rng) -> None:
  r = min(f.L, f.W) / 2
  color = _pick(
    ((0.20, 0.35, 0.60, 1.0), CREAM, TERRACOTTA, (0.85, 0.85, 0.80, 1.0)), rng
  )
  body_h = 0.6 * f.H
  f.box(0, 0, body_h / 2, r * 0.9, r * 0.9, body_h / 2, color, "ellipsoid")
  f.rod((0, 0, body_h * 0.8), (0, 0, f.H * 0.75), r * 0.35, color, "cylinder")
  for k in range(3):  # Stems and flowers.
    ang = k * 2 * np.pi / 3
    tip = (0.4 * r * np.cos(ang), 0.4 * r * np.sin(ang), f.H - r * 0.25)
    f.rod((0, 0, f.H * 0.75), tip, min(0.006, r * 0.05), LEAF_DARK)
    f.sphere(*tip, min(r * 0.25, f.H - tip[2]), _pick(SHIRTS + (CREAM,), rng))


def _lamp(f: _Frame) -> None:
  r = min(f.L, f.W) / 2
  f.rod((0, 0, 0), (0, 0, 0.02), r * 0.6, DARK, "cylinder")
  f.rod((0, 0, 0.02), (0, 0, f.H * 0.75), min(0.012, r * 0.1), METAL, "cylinder")
  f.rod((0, 0, f.H * 0.72), (0, 0, f.H), r * 0.95, CREAM, "cylinder")


def _panel(f: _Frame, frame: Rgba = METAL, posts: bool = True) -> None:
  """A glass pane along the box with a top rail and posts."""
  r = min(0.02, f.W / 2)
  f.slab(-f.L / 2 + r, f.L / 2 - r, -f.W / 2 * 0.4, f.W / 2 * 0.4, 0, f.H - r, GLASS)
  f.rod((-f.L / 2 + r, 0, f.H - r), (f.L / 2 - r, 0, f.H - r), r, frame)
  if posts:
    n = max(1, round(f.L / 1.0))
    for k in range(n + 1):
      u = -f.L / 2 + r + k * (f.L - 2 * r) / n
      f.rod((u, 0, 0), (u, 0, f.H - r), r, frame, "cylinder")


def _mirror(f: _Frame) -> None:
  t = min(0.04, f.W / 2)
  f.slab(-f.L / 2, f.L / 2, -t / 2, t / 2, 0.0, f.H, DARK_WOOD)
  f.slab(-f.L / 2 + 0.04, f.L / 2 - 0.04, t / 2 - 0.004, t / 2, 0.06, f.H - 0.04,
         (0.82, 0.88, 0.92, 1.0))  # fmt: skip


def _tv(f: _Frame) -> None:
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0, 0.45 * f.H, DARK_WOOD)
  f.slab(-f.L / 2 + 0.05, f.L / 2 - 0.05, -0.02, 0.02, 0.5 * f.H, f.H, DARK)
  f.rod((0, 0, 0.45 * f.H), (0, 0, 0.5 * f.H), 0.03, METAL, "cylinder")


def _carton(f: _Frame) -> None:
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0, f.H, CARDBOARD)
  f.slab(-f.L / 2, f.L / 2, -0.03, 0.03, f.H - 0.003, f.H, TAPE)


def _shed(f: _Frame, glass: bool = False) -> None:
  wall = GLASS if glass else WOOD
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0, 0.85 * f.H, wall)
  f.slab(-f.L / 2, f.L / 2, -f.W / 2, f.W / 2, 0.85 * f.H, f.H,
         GLASS_SHELF if glass else DARK_WOOD)  # fmt: skip


def _grill(f: _Frame) -> None:
  r = min(f.L, f.W) / 2
  for s in (-1, 1):
    f.rod((s * r * 0.6, 0, 0), (s * r * 0.6, 0, f.H * 0.55), 0.015, METAL, "cylinder")
  f.box(0, 0, f.H * 0.7, r, r * 0.9, f.H * 0.18, DARK, "ellipsoid")
  f.rod((0, 0, f.H * 0.88), (0, 0, f.H * 0.95), 0.02, METAL, "cylinder")


def _wheelbarrow(f: _Frame) -> None:
  r = min(0.18, f.H / 2, f.W / 2, 0.2 * f.L)
  f.rod((f.L / 2 - r, -0.03, r), (f.L / 2 - r, 0.03, r), r, DARK, "cylinder")
  f.slab(-f.L / 2 + 0.25 * f.L, f.L / 2 - 2 * r, -f.W / 2, f.W / 2, f.H * 0.45, f.H,
         (0.25, 0.45, 0.30, 1.0))  # fmt: skip
  for s in (-1, 1):
    v = s * (f.W / 2 - 0.03)
    f.rod(
      (-f.L / 2 + 0.02, v, f.H * 0.55), (f.L / 2 - 2 * r, v, f.H * 0.45), 0.015, METAL
    )


def _pouf(f: _Frame, rng) -> None:
  r = min(f.L, f.W) / 2
  f.rod((0, 0, 0), (0, 0, f.H), r, _pick(FABRICS, rng), "cylinder")


_HUMAN = {"person", "guest", "visitor", "receptionist"}
_GLASS_PANEL = {
  "glass", "glass partition", "glass rail", "glass balustrade", "glass patio doors",
}  # fmt: skip


def dress(it: Item, seed: int = 0) -> list[Part]:
  """The parts drawing item ``it`` (a ``person``, ``glass`` or ``box``
  item); ``seed`` varies colours between items of the same label."""
  label = it.label.lower()
  rng = np.random.default_rng(zlib.crc32(f"{label}|{seed}".encode()))
  f = _Frame(it)
  if label == "child":
    _human(f, rng, child=True)
  elif label in _HUMAN or it.cls == "person":
    _human(f, rng)
  elif label in ("sofa",):
    _sofa(f, rng)
  elif label == "bed":
    _bed(f, rng)
  elif label == "chairs":
    _chairs(f, rng)
  elif label in ("glass table",):
    _table(f, GLASS, METAL, thick=0.02)
  elif label in ("table", "dining table", "side table", "drinks table"):
    _table(f, WOOD, DARK_WOOD)
  elif label == "desk":
    _desk(f)
  elif label in ("bar", "reception desk"):
    _counter(f, DARK_WOOD)
  elif label == "bookcase":
    _shelves(f, rng, books=True)
  elif label in ("glass cabinet", "display case", "glass vitrine"):
    _vitrine(f, rng)
  elif label in ("wine rack", "shoe rack"):
    _shelves(f, rng)
  elif label in ("sideboard", "toy chest", "laundry"):
    _cabinet(f, rng)
  elif label in ("plant", "planters", "plant stand"):
    _plant(f, rng)
  elif label == "glass planter":
    _plant(f, rng, pot=GLASS_EDGE)
  elif label == "vase":
    _vase(f, rng)
  elif label in ("lamp", "floor lamp"):
    _lamp(f)
  elif label in ("mirror", "standing mirror"):
    _mirror(f)
  elif label == "tv":
    _tv(f)
  elif label in ("box", "toy box"):
    _carton(f)
  elif label == "shed":
    _shed(f)
  elif label == "greenhouse":
    _shed(f, glass=True)
  elif label == "grill":
    _grill(f)
  elif label == "wheelbarrow":
    _wheelbarrow(f)
  elif label == "pouf":
    _pouf(f, rng)
  elif label == "stair rail":
    _panel(f, frame=DARK_WOOD)
  elif label in _GLASS_PANEL or it.cls == "glass":
    _panel(f)
  else:
    _cabinet(f, rng)
  return f.parts
