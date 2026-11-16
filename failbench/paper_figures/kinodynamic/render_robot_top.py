"""Top-down Go2 render (RGBA, cropped) -> argv[1]. Robot faces +x (right)."""

import sys

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, gaussian_filter

spec = mujoco.MjSpec.from_file("go2_him/robot/go2.xml")
m = spec.compile()
W = H = 900
m.vis.global_.offwidth, m.vis.global_.offheight = W, H
m.vis.quality.offsamples = 16
m.vis.global_.fovy = 15
m.vis.headlight.ambient[:] = [0.55] * 3
m.vis.headlight.diffuse[:] = [0.45] * 3
d = mujoco.MjData(m)
d.qpos[2] = 0.30
d.qpos[3:7] = [1, 0, 0, 0]
STAND = {"hip": 0.0, "thigh": 0.8, "calf": -1.5}
for j in range(1, m.njnt):
  d.qpos[m.jnt_qposadr[j]] = STAND[m.joint(j).name.split("_")[1]]
mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0, 0, 0.2]
cam.distance = 3.2
cam.azimuth, cam.elevation = 90, -89.9  # Looking down; +x points right.
r = mujoco.Renderer(m, H, W)
r.update_scene(d, camera=cam)
img = r.render().astype(float) / 255
r.enable_depth_rendering()
r.update_scene(d, camera=cam)
dep = r.render()
fg = dep < dep.max() * 0.99
a = gaussian_filter(
  binary_erosion(fg & (img.mean(-1) > 0.25), iterations=2).astype(float), 0.6
)
rgba = np.dstack([img, a])
ys, xs = np.nonzero(a > 0.02)
rgba = rgba[ys.min() - 4 : ys.max() + 4, xs.min() - 4 : xs.max() + 4]
plt.imsave(sys.argv[1], rgba)
print(rgba.shape)
