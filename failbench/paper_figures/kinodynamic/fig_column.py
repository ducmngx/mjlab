"""Failure-Conditioned Kinodynamic Model, single-column layout: inputs above the
decoder, per-step outputs to its right.

  uv run python failbench/paper_figures/kinodynamic/fig_column.py [out_dir]
Reads assets/ (sample983.npz, robot_RR.png, robot_top.png); writes
fig_kinodynamic_col.pdf/png to out_dir (default: ./out next to this file).
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
TOP = plt.imread(HERE / "robot_top.png")

T1, T2, T3 = "#111111", "#333333", "#777777"
ARROW = "#444444"
ENC, ENC_E = "#e3eaf5", "#7088b0"
DEC, DEC_E = "#ece8f6", "#8a7bbd"
MEM, MEM_E = "#ddd5f0", "#7a66b5"
FAIL, FAIL_E, FAIL_D = "#fde1c8", "#de7f2e", "#d0342c"
QRY, QRY_E = "#d4ecdc", "#3e9c63"
OUT_E, OUT_D = "#3d7cc9", "#1c4a86"
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
    "savefig.dpi": 500,
    "text.color": T2,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
  }
)
FW, FH = 3.4, 1.63
fig = plt.figure(figsize=(FW, FH))
ax = fig.add_axes((0, 0, 1, 1))
ax.set_xlim(0, FW * 100)
ax.set_ylim(61, 61 + FH * 100)
ax.set_axis_off()


def rbox(x0, y0, w, h, fc="white", ec="#999999", lw=0.7, r=3.0, z=1.0):
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


def label(x, y, s, fs=6.5, c=T2, ha="center", va="center", w="normal", z=10, **kw):
  return ax.text(
    x, y, s, fontsize=fs, color=c, ha=ha, va=va, fontweight=w, zorder=z, **kw
  )


def arrow(p, q, c=ARROW, lw=0.7, ms=4.5, z=4):
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


def tokens(cx, y, fcs, ec, w=6.5, h=9, gap=2.2):
  n = len(fcs)
  x0 = cx - (n * w + (n - 1) * gap) / 2
  for i, fc in enumerate(fcs):
    rbox(x0 + i * (w + gap), y - h / 2, w, h, fc=fc, ec=ec, lw=0.5, r=1.5, z=6)


def image(arr, x0, y0, w, h, z=3, r=2.5, **kw):
  art = ax.imshow(arr, extent=(x0, x0 + w, y0, y0 + h), zorder=z, aspect="auto", **kw)
  clip = rbox(x0, y0, w, h, fc="none", ec="#a0a0a0", lw=0.5, r=r, z=z + 0.5)
  art.set_clip_path(clip)


def shaded(hmap):
  t = np.clip((hmap + 0.4) / 0.95, 0, 1)
  return LightSource(azdeg=315, altdeg=45).shade(
    t, cmap=RAMP, blend_mode="soft", vert_exag=1.5
  )[..., :3]


def marker(x, y, s=1.0, z=8):
  pts = np.array([[4.6, 0], [-2.8, 3.0], [-1.3, 0], [-2.8, -3.0]]) * s
  ax.add_patch(Polygon(pts + (x, y), fc=RED, ec="white", lw=0.4, zorder=z))


def go2(x, y, yaw_deg=0.0, length=30, alpha=1.0, z=6, tint=None):
  pre = TOP.copy()
  pre[..., :3] *= pre[..., 3:]
  im = np.clip(
    rotate(pre, yaw_deg, reshape=True, order=1, mode="constant", cval=0.0), 0, 1
  )
  im[..., :3] /= np.maximum(im[..., 3:], 1e-3)
  im = np.clip(im, 0, 1)
  if tint is not None:
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


# Geometry: three input columns over one decoder; outputs to the right.
CAP, IMG0, IMG1 = 215, 166, 206  # Caption line, image band.
TOKY = 113
DX0, DX1, DY0, DY1 = 6, 224, 68, 99
CX = (6, 82)  # Context column.
FXc = 120  # Failure column centre.
PX = (154, 224)  # Plan column.

# --- Context: terrain + state history -> encoder -> memory ------------------------
label(CX[0], CAP, "robot state", fs=7, c=T1, ha="left")
# Terrain | proprioception history | policy latent, side by side.
mw = 31
image(
  shaded(S["map"]),
  CX[0],
  IMG0,
  mw,
  IMG1 - IMG0,
  origin="lower",
  interpolation="nearest",
)
marker(CX[0] + mw / 2, (IMG0 + IMG1) / 2, s=0.55)
label(CX[0] + mw / 2, IMG0 - 5, "terrain", fs=5.6, c=T3)
hist = S["history"]  # [20, 52]: state (33), command (3), latent (16).
px0, pw = CX[0] + mw + 3, 30
for k in (2, 1):  # Past frames: the history window behind the current state.
  rbox(
    px0 + 2.2 * k,
    IMG0 + 2.2 * k,
    pw,
    IMG1 - IMG0,
    fc="white",
    ec="#c8c8c8",
    lw=0.45,
    r=2.5,
    z=2.6 - 0.1 * k,
  )
rbox(px0, IMG0, pw, IMG1 - IMG0, fc="white", ec="#a0a0a0", lw=0.5, r=2.5, z=3)
tt = np.linspace(0, 1, hist.shape[0])
tf = np.linspace(0, 1, 120)
band = (IMG1 - IMG0 - 6) / 4
for n, (j, c) in enumerate(
  zip((10, 13, 16, 19), ("#3d7cc9", "#3e9c63", "#de7f2e", "#8a7bbd"), strict=True)
):
  y = hist[:, j]  # Thigh joint positions over the history window.
  y = (y - y.mean()) / (np.ptp(y) + 1e-6)
  ax.plot(
    px0 + 2.5 + (pw - 5) * tf,
    IMG1 - 3 - band * (n + 0.5) + np.interp(tf, tt, y) * band * 0.9,
    color=c,
    lw=0.65,
    zorder=4,
  )
# "Now": the current state s_tau closes the window.
ax.plot(
  [px0 + pw - 2.5] * 2,
  [IMG0 + 2, IMG1 - 2],
  color=T3,
  lw=0.5,
  ls=(0, (1.2, 1.0)),
  zorder=4,
)
label(px0 + pw / 2 + 8, IMG0 - 5, "state, history", fs=5.6, c=T3)
# Policy latent z: a column of cells.
lat = S["latent"]
lx0, lw_ = px0 + pw + 7, 6
LAT = LinearSegmentedColormap.from_list("lat", ["#f3f0fa", "#b9aee0", "#6f5bb0"])
lz = np.argsort(np.argsort(lat)) / (len(lat) - 1)  # Rank: shows every cell.
image(
  lz[::-1, None],
  lx0,
  IMG0,
  lw_,
  IMG1 - IMG0,
  cmap=LAT,
  vmin=0,
  vmax=1,
  interpolation="nearest",
  r=1.5,
)
label(lx0 + lw_ + 3.5, (IMG0 + IMG1) / 2, "internal", fs=5.6, c=T3, rotation=90)
ccx = (CX[0] + CX[1]) / 2
ey0, ey1 = 128, 149
ax.add_patch(
  Polygon(
    [(CX[0] + 2, ey1), (CX[1] - 2, ey1), (ccx + 20, ey0), (ccx - 20, ey0)],
    fc=ENC,
    ec=ENC_E,
    lw=0.7,
    joinstyle="round",
    zorder=3,
  )
)
label(ccx, (ey0 + ey1) / 2 + 0.5, "Context\nEncoder", fs=6.2, c=T1, linespacing=0.95)
arrow((ccx, IMG0 - 10), (ccx, ey1), ms=4)
arrow((ccx, ey0), (ccx, TOKY + 5), ms=4)
tokens(ccx, TOKY, [MEM] * 5, MEM_E)
label(ccx + 24, TOKY, "$c_\\tau$", fs=7, c=MEM_E, ha="left")

# --- Failure: robot with the failed leg in red -> joint tokens --------------------
label(FXc - 22, CAP, "failure  $f$", fs=7, c=T1, ha="left")
rob = plt.imread(HERE / "robot_RR.png")
rh = 46
rw = rh * rob.shape[1] / rob.shape[0]
ax.imshow(
  rob,
  extent=(FXc - rw / 2, FXc + rw / 2, IMG0 - 6, IMG0 - 6 + rh),
  zorder=4,
  interpolation="lanczos",
)
arrow((FXc, IMG0 - 8), (FXc, TOKY + 5), ms=4)
tokens(FXc, TOKY, [FAIL, FAIL, FAIL_D, FAIL, FAIL], FAIL_E)

# --- Plan: commanded path x̂ with commands u along it -> queries -------------------
label(PX[0], CAP, "plan", fs=7, c=T1, ha="left")
cs, cp = S["cmd_seq"], S["cmd_path"]
H = len(cp)
sc = (PX[1] - PX[0] - 26) / np.ptp(cp[:, 0])
ox = PX[0] + 16
oy = 182 - (cp[:, 1].max() + cp[:, 1].min()) / 2 * sc
px, py = ox + cp[:, 0] * sc, oy + cp[:, 1] * sc
go2(
  ox - 2,
  oy,
  np.degrees(np.arctan2(py[2] - py[0], px[2] - px[0])),
  length=18,
  tint=QRY_E,
  z=6,
)
ax.plot(px, py, color=QRY_E, lw=1.0, alpha=0.55, zorder=3)
for i in (8, 17, 26):
  ang = np.arctan2(py[min(i + 1, H - 1)] - py[i - 1], px[min(i + 1, H - 1)] - px[i - 1])
  L = 12 * cs[i, 0]
  c0, s0 = np.cos(ang), np.sin(ang)
  arrow(
    (px[i] + 2 * c0, py[i] + 2 * s0),
    (px[i] + L * c0, py[i] + L * s0),
    c=QRY_E,
    lw=0.9,
    ms=4,
    z=5,
  )
  ax.add_patch(Circle((px[i], py[i]), 1.5, fc="white", ec=QRY_E, lw=0.8, zorder=6))
label(px[17] + 4, py[17] + 7, "$u_k$", fs=6.3, c=QRY_E)
label(px[17] - 1, py[17] - 8, "$\\hat{x}_k$", fs=6.3, c=QRY_E)
pcx = (PX[0] + PX[1]) / 2
arrow((pcx, IMG0 - 2), (pcx, TOKY + 5), ms=4)
tokens(pcx, TOKY, [QRY] * 5, QRY_E)

# --- Decoder ------------------------------------------------------------------------
rbox(DX0, DY0, DX1 - DX0, DY1 - DY0, fc=DEC, ec=DEC_E, lw=0.8, r=5, z=2)
label((DX0 + DX1) / 2, (DY0 + DY1) / 2, "Transformer Decoder", fs=7.6, c=T1, w="bold")
for cx_ in (ccx, FXc, pcx):
  arrow((cx_, TOKY - 5), (cx_, DY1), ms=4)

# --- Outputs: drift and velocity cards to the right --------------------------------
TX0, TW, TH = 240, 92, 52
DYc, VYc = 168, 99


def card(y0, title):
  for k in (2, 1):
    rbox(
      TX0 + 3 * k,
      y0 + 3 * k,
      TW - 6,
      TH,
      fc="white",
      ec="#c4c4c4",
      lw=0.5,
      r=3,
      z=1 + 0.1 * (3 - k),
    )
  rbox(TX0, y0, TW - 6, TH, fc="white", ec="#9a9a9a", lw=0.6, r=3, z=1.5)
  label(TX0 + 5, y0 + TH - 7, title, fs=6.6, c=T1, ha="left")


SPLIT = DX1 + 7
ax.plot([DX1, SPLIT], [(DY0 + DY1) / 2] * 2, color=ARROW, lw=0.7, zorder=4)
ax.plot([SPLIT, SPLIT], [min(VYc, (DY0 + DY1) / 2), DYc], color=ARROW, lw=0.7, zorder=4)
for yc in (DYc, VYc):
  arrow((SPLIT, yc), (TX0, yc), ms=4)

y0 = DYc - TH / 2
card(y0, "drift  $\\delta_k$")
xx = np.linspace(TX0 + 5, TX0 + TW - 11, 40)
ax.plot(
  xx,
  y0 + 30 + 3 * np.sin((xx - TX0) / TW * 2.2),
  color=QRY_E,
  lw=1.0,
  ls=(0, (3, 1.8)),
  zorder=3,
)
gx = TX0 + 62
gy = y0 + 30 + 3 * np.sin(62 / TW * 2.2)
go2(gx, gy, 8, length=28, alpha=0.55, tint=QRY_E)
ax_, ay_ = TX0 + 32, y0 + 13
for fc, ec in ((OUT_E, "none"), ("none", OUT_E)):
  ax.add_patch(
    Ellipse(
      (ax_, ay_),
      38,
      17,
      angle=-12,
      fc=fc,
      ec=ec,
      lw=0.5,
      alpha=0.15 if fc != "none" else 0.7,
      zorder=3,
    )
  )
go2(ax_, ay_, -24, length=28, z=7, tint=OUT_E)
arrow((gx, gy), (ax_ + 2, ay_ + 1), c=T1, lw=0.8, ms=4.5, z=8)

y0 = VYc - TH / 2
card(y0, "velocity  $v_k$")
rx, ry, ang = TX0 + 28, y0 + 20, -15
ax.add_patch(
  Wedge((rx, ry), 40, ang - 16, ang + 16, fc=OUT_E, ec="none", alpha=0.15, zorder=3)
)
go2(rx, ry, ang, length=28, z=6, tint=OUT_E)
arrow(
  (rx + 5, ry - 1.3),
  (rx + 38 * np.cos(np.radians(ang)), ry + 38 * np.sin(np.radians(ang))),
  c=OUT_D,
  lw=1.2,
  ms=6,
  z=8,
)

for ext in ("pdf", "png"):
  fig.savefig(OUT / f"fig_kinodynamic_col.{ext}", pad_inches=0.0)
print("saved", OUT)
