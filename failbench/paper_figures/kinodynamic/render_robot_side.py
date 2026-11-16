"""Render the Go2 with the failed leg tinted red -> robot_<leg>.png (RGBA)."""

import os
import sys

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter

OUT, LEG = sys.argv[1], sys.argv[2]  # e.g. RR
AZ, EL = float(os.environ.get("AZ", 40)), float(os.environ.get("EL", -18))
spec = mujoco.MjSpec.from_file("go2_him/robot/go2.xml")
spec.worldbody.add_light(pos=[0.5, -1, 2.5], dir=[-0.2, 0.4, -1], diffuse=[0.6] * 3)
m = spec.compile()
W, H = 1200, 900
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
m.vis.quality.offsamples = 16
m.vis.global_.fovy = 18
m.vis.headlight.ambient[:] = [0.45] * 3
m.vis.headlight.diffuse[:] = [0.45] * 3
d = mujoco.MjData(m)
d.qpos[2] = 0.30
d.qpos[3:7] = [1, 0, 0, 0]
STAND = {"hip": 0.0, "thigh": 0.8, "calf": -1.5}
for j in range(1, m.njnt):
  d.qpos[m.jnt_qposadr[j]] = STAND[m.joint(j).name.split("_")[1]]
mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.22]
cam.distance = 2.6
cam.azimuth, cam.elevation = AZ, EL
r = mujoco.Renderer(m, H, W)


def shot(rgba=None):
  if rgba is not None:
    m.geom_rgba[:] = rgba
  r.update_scene(d, camera=cam)
  return r.render().astype(float) / 255


base_rgba = m.geom_rgba.copy()
img = shot()
r.enable_depth_rendering()
r.update_scene(d, camera=cam)
dep = r.render()
r.disable_depth_rendering()
fg = dep < dep.max() * 0.99
bodies = [m.body(f"{LEG}_{s}").id for s in ("thigh", "calf")]
leg = np.isin(m.geom_bodyid, bodies)
mat = m.geom_matid.copy()
rg = base_rgba.copy()
rg[:] = [0, 0, 0, 1]
rg[leg] = [1, 1, 1, 1]
m.geom_matid[:] = -1
mask = shot(rg).mean(-1) > 0.5
m.geom_matid[:] = mat
m.geom_rgba[:] = base_rgba
red = np.array([0.86, 0.16, 0.13])
soft = gaussian_filter(mask.astype(float), 1.0)[..., None]
lum = img.mean(-1, keepdims=True)
tint = 0.35 * img + 0.65 * red * (0.55 + 0.6 * lum)
out = img * (1 - soft * 0.9) + tint * soft * 0.9
# Robot layer over a soft red glow around the failed leg.
ar = gaussian_filter(binary_erosion(fg, iterations=2).astype(float), 0.7)
ring = gaussian_filter(binary_dilation(mask, iterations=8).astype(float), 3.5)
ag = 0.6 * ring * (1 - ar)
A = ar + ag
C = (out * ar[..., None] + red * ag[..., None]) / np.maximum(A, 1e-6)[..., None]
rgba = np.dstack([np.clip(C, 0, 1), np.clip(A, 0, 1)])
a = A
ys, xs = np.nonzero(a > 0.02)
rgba = rgba[ys.min() - 6 : ys.max() + 6, xs.min() - 6 : xs.max() + 6]
plt.imsave(OUT, rgba)
print("saved", OUT, rgba.shape)
