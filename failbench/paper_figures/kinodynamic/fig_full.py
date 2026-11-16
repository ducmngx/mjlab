"""Failure-Conditioned Kinodynamic Model: full-width paper figure (one real sample).

  uv run python failbench/paper_figures/kinodynamic/fig_full.py [out_dir]
Reads assets/ (sample983.npz, robot_RR.png, robot_top.png); writes
fig_kinodynamic.pdf/png to out_dir (default: ./out next to this file).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, to_rgb
from matplotlib.patches import (
  Circle,
  Ellipse,
  FancyArrowPatch,
  FancyBboxPatch,
  Polygon,
  Wedge,
)
from scipy.ndimage import rotate

HERE = Path(__file__).parent / "assets"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)
S = np.load(HERE / "sample983.npz")

T1, T2, T3 = "#111111", "#333333", "#777777"
ARROW = "#444444"
ENC, ENC_E = "#e3eaf5", "#7088b0"
DEC, DEC_E = "#ece8f6", "#8a7bbd"
MEM, MEM_E = "#ddd5f0", "#7a66b5"
FAIL, FAIL_E, FAIL_D = "#fde1c8", "#de7f2e", "#d0342c"
QRY, QRY_E = "#d4ecdc", "#3e9c63"
OUTC, OUT_E, OUT_D = "#d4e4f8", "#3d7cc9", "#1c4a86"
RED = "#d0342c"
DIV = LinearSegmentedColormap.from_list(
  "div", ["#2f6db5", "#9cc3ea", "#f7f7f5", "#f2b27e", "#d0612a"]
)
RAMP = LinearSegmentedColormap.from_list(
  "h", ["#8a7656", "#a8916c", "#c2b08f", "#dacdb5", "#efe9dd"]
)

plt.rcParams.update(
  {
    "font.family": "Arial",
    "font.size": 7,
    "pdf.fonttype": 42,
    "savefig.dpi": 400,
    "text.color": T2,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
  }
)
FW, FH = 4.86, 2.2
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, FW * 100)
ax.set_ylim(0, FH * 100)
ax.set_axis_off()


def rbox(x0, y0, w, h, fc="white", ec="#999999", lw=0.8, r=4.0, z=1.0):
  p = FancyBboxPatch(
    (x0, y0),
    w,
    h,
    boxstyle=f"round,pad=0,rounding_size={r}",
    fc=fc,
    ec=ec,
    lw=lw,
    zorder=z,
  )
  ax.add_patch(p)
  return p


def label(x, y, s, fs=7.0, c=T2, ha="center", va="center", w="normal", z=10, **kw):
  return ax.text(
    x, y, s, fontsize=fs, color=c, ha=ha, va=va, fontweight=w, zorder=z, **kw
  )


def arrow(p, q, c=ARROW, lw=0.8, ms=5.5, z=4):
  ax.add_patch(
    FancyArrowPatch(
      p,
      q,
      arrowstyle="-|>",
      mutation_scale=ms,
      color=c,
      lw=lw,
      shrinkA=0,
      shrinkB=0,
      zorder=z,
    )
  )


def tokens(x0, y, fcs, ec, w=8, h=11, gap=3):
  for i, fc in enumerate(fcs):
    rbox(x0 + i * (w + gap), y - h / 2, w, h, fc=fc, ec=ec, lw=0.6, r=1.8, z=6)
  return x0 + len(fcs) * (w + gap) - gap


def image(arr, x0, y0, w, h, z=3, **kw):
  art = ax.imshow(arr, extent=(x0, x0 + w, y0, y0 + h), zorder=z, aspect="auto", **kw)
  clip = rbox(x0, y0, w, h, fc="none", ec="#a0a0a0", lw=0.6, r=3, z=z + 0.5)
  art.set_clip_path(clip)
  return art


def shaded(hmap):
  t = np.clip((hmap + 0.4) / 0.95, 0, 1)
  return LightSource(azdeg=315, altdeg=45).shade(
    t, cmap=RAMP, blend_mode="soft", vert_exag=1.5
  )[..., :3]


def marker(x, y, s=1.0, z=8):
  pts = np.array([[4.6, 0], [-2.8, 3.0], [-1.3, 0], [-2.8, -3.0]]) * s
  ax.add_patch(Polygon(pts + (x, y), fc=RED, ec="white", lw=0.5, zorder=z))


def caption(x, y, s):
  label(x, y, s, fs=7, c=T1, ha="left")


# Rows: context, failure, plan.
YC, YF, YP = 166, 96, 34
IX = 10

# --- Context: terrain map + state history -> encoder -> memory ---------------------
caption(IX, 205, "context  $c_\\tau$")
image(shaded(S["map"]), IX, YC - 26, 52, 52, origin="lower", interpolation="nearest")
marker(IX + 26, YC, s=0.8)
hist = S["history"]
z = (hist - hist.mean(0)) / (hist.std(0) + 1e-3)
image(
  z.T[::-1],
  IX + 58,
  YC - 26,
  40,
  52,
  cmap=DIV,
  vmin=-2.2,
  vmax=2.2,
  interpolation="nearest",
)
label(IX + 26, YC - 33, "terrain", fs=6.3, c=T3)
label(IX + 78, YC - 33, "state history", fs=6.3, c=T3)
EX0, EX1 = 122, 150
ax.add_patch(
  Polygon(
    [(EX0, YC - 30), (EX1, YC - 18), (EX1, YC + 18), (EX0, YC + 30)],
    fc=ENC,
    ec=ENC_E,
    lw=0.8,
    joinstyle="round",
    zorder=3,
  )
)
label((EX0 + EX1) / 2, YC, "Encoder", fs=7, c=T1, rotation=90)
arrow((IX + 100, YC), (EX0, YC))
TX = 164
arrow((EX1, YC), (TX - 3, YC))
tx1 = tokens(TX, YC, [MEM] * 5, MEM_E)
label((TX + tx1) / 2, YC + 12, "memory", fs=6.3, c=MEM_E)

# --- Failure: robot with the failed leg in red -> joint tokens ---------------------
caption(IX, YF + 21, "failure  $f$")
rob = plt.imread(HERE / "robot_RR.png")
rh = 44
rw = rh * rob.shape[1] / rob.shape[0]
ax.imshow(
  rob,
  extent=(IX + 52 - rw / 2, IX + 52 + rw / 2, YF - 24, YF - 24 + rh),
  zorder=4,
  interpolation="lanczos",
)
arrow((IX + 100, YF), (TX - 3, YF))
fx1 = tokens(TX, YF, [FAIL, FAIL, FAIL_D, FAIL, FAIL], FAIL_E)
label((TX + fx1) / 2, YF + 12, "per joint", fs=6.3, c=FAIL_E)

TOP = plt.imread(HERE / "robot_top.png")


def go2(x, y, yaw_deg=0.0, length=36, alpha=1.0, z=6, tint=None):
  pre = TOP.copy()
  pre[..., :3] *= pre[..., 3:]  # Rotate premultiplied, so edges stay clean.
  im = np.clip(
    rotate(pre, yaw_deg, reshape=True, order=1, mode="constant", cval=0.0), 0, 1
  )
  im[..., :3] /= np.maximum(im[..., 3:], 1e-3)
  im = np.clip(im, 0, 1)
  if tint is not None:  # Shade the render toward a colour, keeping its detail.
    lum = im[..., :3].mean(-1, keepdims=True)
    col = np.array(to_rgb(tint)) * (0.45 + 0.7 * lum)
    im[..., :3] = np.clip(0.45 * im[..., :3] + 0.55 * col, 0, 1)
  im[..., 3] *= alpha
  s = length / TOP.shape[1]
  w, h = np.size(im, 1) * s, np.size(im, 0) * s
  ax.imshow(
    im,
    extent=(x - w / 2, x + w / 2, y - h / 2, y + h / 2),
    zorder=z,
    interpolation="lanczos",
  )


# --- Plan: the commanded path x̂ with the commands u along it -> step queries -----
caption(IX, YP + 30, "plan")
cs, cp = S["cmd_seq"], S["cmd_path"]
H = len(cp)
sc = 72 / np.ptp(cp[:, 0])
ox = IX + 18
oy = YP - (cp[:, 1].max() + cp[:, 1].min()) / 2 * sc
px, py = ox + cp[:, 0] * sc, oy + cp[:, 1] * sc
go2(
  ox - 2,
  oy,
  np.degrees(np.arctan2(py[2] - py[0], px[2] - px[0])),
  length=24,
  tint=QRY_E,
  z=6,
)
ax.plot(px, py, color=QRY_E, lw=1.2, alpha=0.55, zorder=3)
for i in (7, 15, 23):
  ang = np.arctan2(py[min(i + 1, H - 1)] - py[i - 1], px[min(i + 1, H - 1)] - px[i - 1])
  L = 19 * cs[i, 0]
  c0, s0 = np.cos(ang), np.sin(ang)
  arrow(
    (px[i] + 2.5 * c0, py[i] + 2.5 * s0),
    (px[i] + L * c0, py[i] + L * s0),
    c=QRY_E,
    lw=1.1,
    ms=5,
    z=5,
  )
  ax.add_patch(Circle((px[i], py[i]), 1.8, fc="white", ec=QRY_E, lw=0.9, zorder=6))
i = 15
label(px[i] + 6, py[i] + 8, "$u_k$", fs=6.8, c=QRY_E)
label(px[i] - 1, py[i] - 9, "$\\hat{x}_k$", fs=6.8, c=QRY_E)
arrow((IX + 104, YP), (TX - 3, YP))
qx1 = tokens(TX, YP, [QRY] * 5, QRY_E)
label((TX + qx1) / 2, YP + 12, "$(u_k, \\hat{x}_k)$", fs=6.5, c=QRY_E)

# --- Decoder ------------------------------------------------------------------------
DX0, DX1, DY0, DY1 = 236, 322, 22, 178
rbox(DX0, DY0, DX1 - DX0, DY1 - DY0, fc=DEC, ec=DEC_E, lw=0.9, r=6, z=2)
label(
  (DX0 + DX1) / 2,
  (DY0 + DY1) / 2 + 5,
  "Transformer\nDecoder",
  fs=8,
  c=T1,
  w="bold",
  linespacing=1.1,
)
label((DX0 + DX1) / 2, (DY0 + DY1) / 2 - 15, "causal over steps", fs=6.3, c=T3)
for y, x1 in ((YC, tx1), (YF, fx1), (YP, qx1)):
  arrow((x1 + 2, y), (DX0, y))
label(DX0 - 7, YC - 10, "keys, values", fs=6.0, c=T3, ha="right")
label(DX0 - 7, YP - 10, "queries", fs=6.0, c=T3, ha="right")


# --- Outputs: per-step drift and velocity, as pictures ------------------------------
def card(x0, y0, w, h, title):
  for k in (2, 1):  # One card per step: a short stack behind.
    rbox(
      x0 + 3.5 * k,
      y0 + 3.5 * k,
      w,
      h,
      fc="white",
      ec="#c4c4c4",
      lw=0.6,
      r=4,
      z=1 + 0.1 * (3 - k),
    )
  rbox(x0, y0, w, h, fc="white", ec="#9a9a9a", lw=0.7, r=4, z=1.5)
  label(x0 + 6, y0 + h - 8, title, fs=7, c=T1, ha="left")


TW, TH = 104, 62
OYm = (DY0 + DY1) / 2
TX0 = DX1 + 22
DYc, VYc = OYm + 38, OYm - 40
for yc in (DYc, VYc):
  arrow((DX1, yc), (TX0, yc))

# Drift: where the robot ends up vs where the plan puts it, with spread.
y0 = DYc - TH / 2
card(TX0, y0, TW, TH, "drift  $\\delta_k$")
xx = np.linspace(TX0 + 6, TX0 + TW - 6, 50)
ax.plot(
  xx,
  y0 + 34 + 4 * np.sin((xx - TX0) / TW * 2.2),
  color=QRY_E,
  lw=1.2,
  ls=(0, (3.5, 2)),
  zorder=3,
)
gx, gy = TX0 + 74, y0 + 34 + 4 * np.sin(74 / TW * 2.2)
go2(gx, gy, 8, alpha=0.55, tint=QRY_E)
ax_, ay_ = TX0 + 38, y0 + 16
ax.add_patch(
  Ellipse(
    (ax_, ay_), 52, 26, angle=-12, fc=OUT_E, ec=OUT_E, lw=0.6, alpha=0.15, zorder=3
  )
)
ax.add_patch(
  Ellipse(
    (ax_, ay_), 52, 26, angle=-12, fc="none", ec=OUT_E, lw=0.6, alpha=0.7, zorder=3
  )
)
go2(ax_, ay_, -24, alpha=1.0, z=7, tint=OUT_E)
arrow((gx, gy), (ax_ + 2, ay_ + 1), c=T1, lw=0.9, ms=5.5, z=8)
label(TX0 + TW - 6, y0 + 12, "$\\hat{x}_k \\to x_k$", fs=6.3, c=T3, ha="right")

# Velocity: the trunk's velocity, with spread.
y0 = VYc - TH / 2
card(TX0, y0, TW, TH, "velocity  $v_k$")
rx, ry = TX0 + 34, y0 + 25
ang = -15
ax.add_patch(
  Wedge((rx, ry), 46, ang - 16, ang + 16, fc=OUT_E, ec="none", alpha=0.15, zorder=3)
)
go2(rx, ry, ang, z=6, tint=OUT_E)
arrow(
  (rx + 6, ry - 1.6),
  (rx + 44 * np.cos(np.radians(ang)), ry + 44 * np.sin(np.radians(ang))),
  c=OUT_D,
  lw=1.4,
  ms=7,
  z=8,
)
label(
  TX0 + TW / 2,
  y0 - 12,
  "per step $k$:  mean $\\mu_k$, spread $\\sigma_k$",
  fs=6.3,
  c=T3,
)

for ext in ("pdf", "png"):
  fig.savefig(OUT / f"fig_kinodynamic.{ext}", pad_inches=0.0)
print("saved", OUT)
