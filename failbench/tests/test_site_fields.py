import math

import mujoco
import numpy as np
import pytest
import torch
from failbench.mppi.cbf import fields_of_levels, severity_levels
from failbench.mppi.costmap import FootprintCfg
from failbench.mppi.site_fields import SiteFields, scan_site

RES = 0.05
SCENE = """
<mujoco>
  <worldbody>
    <geom name="terrain_floor" type="plane" size="6 6 0.1"/>
    <geom name="course_box_0" type="box" size="0.3 0.2 0.4" pos="0.613 0.287 0.4"/>
    <geom name="course_wall_0" type="box" size="0.05 1.0 0.5" pos="-0.812 0 0.5"/>
    <body name="robot" pos="0 0 0.3">
      <freejoint/>
      <geom type="sphere" size="0.1" group="2"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def model():
  return mujoco.MjModel.from_xml_string(SCENE)


def _ego_scan(model, pos, yaw, cells, z0):
  """Height [1, cells, cells] (relative to pos z) and geom ids of the
  egocentric, yaw-aligned grid, rows along y, as the runner's scanner."""
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  half = (cells - 1) / 2
  h = np.full((cells, cells), np.nan)
  ids = np.full((cells, cells), -1)
  gid = np.zeros(1, np.int32)
  c, s = math.cos(yaw), math.sin(yaw)
  for j in range(cells):
    for i in range(cells):
      x, y = (i - half) * RES, (j - half) * RES
      p = np.array([pos[0] + c * x - s * y, pos[1] + s * x + c * y, z0])
      d = mujoco.mj_ray(
        model,
        data,
        p,
        np.array([0, 0, -1.0]),
        np.array([1, 0, 0, 0, 0, 0], np.uint8),
        True,
        -1,
        gid,
      )
      if d >= 0:
        h[j, i] = z0 - d - pos[2]
        ids[j, i] = gid[0]
  return torch.tensor(h, dtype=torch.float32)[None], torch.tensor(ids)[None]


def _sev_of(model):
  sev = torch.zeros(model.ngeom + 1)
  sev[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "course_box_0")] = 8.0
  sev[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "course_wall_0")] = 4.0
  return sev


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2])
def test_prebuilt_fields_match_per_cycle_fields(model, yaw):
  """Away from the window's edge, where the per-cycle map can't see what
  lies beyond, the prebuilt site fields cropped at the robot's pose equal
  the fields built from the robot's own scan."""
  fp = FootprintCfg()
  cells, pos, z0 = 81, (0.1, -0.05, 0.3), 2.3
  sev = _sev_of(model)
  site = scan_site(model, (-3.0, -3.0, 3.0, 3.0), RES, z0)
  fields = SiteFields(site, sev[None], RES, fp)
  got = fields.sample(torch.tensor([pos[:2]]), torch.tensor([yaw]), cells)
  h, ids = _ego_scan(model, pos, yaw, cells, z0)
  severity = sev[ids % len(sev)]
  want = fields_of_levels(severity_levels(h, severity, RES, fp), RES, fp)
  assert sorted(v for v, _ in got) == sorted(v for v, _ in want)
  edge = math.ceil(fp.max_dist_m / RES) + 2
  inner = (slice(None), slice(edge, cells - edge), slice(edge, cells - edge))
  for (v, g), (_, w) in zip(sorted(got), sorted(want), strict=True):
    assert torch.allclose(g[inner], w[inner], atol=1e-4), v


def test_rotated_crop_is_close(model):
  """At any heading, bilinear resampling and the 8-neighbour distance on the turned
  grid keep the distance within about a cell."""
  fp = FootprintCfg()
  cells, pos, yaw, z0 = 81, (0.07, 0.02), 0.6, 2.3
  sev = _sev_of(model)
  site = scan_site(model, (-3.0, -3.0, 3.0, 3.0), RES, z0)
  got = dict(
    SiteFields(site, sev[None], RES, fp).sample(
      torch.tensor([pos[:2]]), torch.tensor([yaw]), cells
    )
  )
  h, ids = _ego_scan(model, (*pos[:2], 0.3), yaw, cells, z0)
  severity = sev[ids % len(sev)]
  want = dict(fields_of_levels(severity_levels(h, severity, RES, fp), RES, fp))
  edge = math.ceil(fp.max_dist_m / RES) + 2
  for v, w in want.items():
    g = got[v][:, edge:-edge, edge:-edge]
    assert (g - w[:, edge:-edge, edge:-edge]).abs().max() <= 1.5 * RES, v


def test_out_of_view_levels_are_dropped(model):
  """A level with nothing within reach of the window isn't listed, as the
  per-cycle fields list only the levels in view."""
  fp = FootprintCfg()
  sev = _sev_of(model)
  site = scan_site(model, (-6.0, -6.0, 6.0, 6.0), RES, 2.3)
  got = SiteFields(site, sev[None], RES, fp).sample(
    torch.tensor([[4.5, 4.5]]), torch.tensor([0.0]), 41
  )
  assert got == []
