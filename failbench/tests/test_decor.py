import mujoco
import numpy as np
import pytest
from failbench.decor import dress, part_bounds
from failbench.house import LAYOUTS
from failbench.house_terrain import HouseTerrainCfg


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_dressing_stays_inside_the_box(layout):
  """What is drawn never reaches past the box the robot can hit, so the
  scene looks like what the checks see."""
  for it in LAYOUTS[layout].items:
    if it.cls in ("floor", "wall"):
      continue
    parts = dress(it, seed=0)
    assert parts, it
    for p in parts:
      assert min(p.b if p.kind in ("box", "ellipsoid") else (p.r,)) > 0, (it, p)
      lo, hi = part_bounds(p)
      assert lo[0] >= it.x0 - 1e-6 and hi[0] <= it.x1 + 1e-6, (it, p)
      assert lo[1] >= it.y0 - 1e-6 and hi[1] <= it.y1 + 1e-6, (it, p)
      assert lo[2] >= it.z0 - 1e-6 and hi[2] <= it.z1 + 1e-6, (it, p)


OFFSET = np.array([7.3, -4.1, 0.0])
"""Where the tile lands, as ``TerrainGenerator`` moves a tile's geoms."""


def _build(layout: str, decor: bool) -> mujoco.MjModel:
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  out = HouseTerrainCfg(layout=layout, decor=decor).function(
    0.0, spec, np.random.default_rng(0)
  )
  for tg in out.geometries:  # As the generator places a tile.
    if tg.geom is not None:
      tg.geom.pos = np.array(tg.geom.pos) + OFFSET
  return spec.compile()


@pytest.mark.parametrize("layout", ["apartment", "garden"])
def test_decor_moves_with_its_boxes(layout):
  """Placed anywhere, every figure part stands inside one of the hidden
  boxes it draws."""
  m = _build(layout, True)
  data = mujoco.MjData(m)
  mujoco.mj_forward(m, data)
  hidden = [
    g
    for g in range(m.ngeom)
    if m.geom(g).name.startswith("course_") and m.geom_rgba[g][3] < 0.01
  ]
  lo = np.array([data.geom_xpos[g] - m.geom_size[g] for g in hidden])
  hi = np.array([data.geom_xpos[g] + m.geom_size[g] for g in hidden])
  for g in range(m.ngeom):
    if m.geom(g).name.startswith("decor_"):
      c = data.geom_xpos[g]
      inside = ((c >= lo - 1e-6) & (c <= hi + 1e-6)).all(-1)
      assert inside.any(), (m.geom(g).name, c)


def _course(m: mujoco.MjModel) -> list[tuple]:
  out = []
  for g in range(m.ngeom):
    name = m.geom(g).name
    if name.startswith("course_"):
      out.append(
        (
          name.rsplit("_", 1)[0],
          tuple(np.round(m.geom_size[g], 6)),
          tuple(np.round(m.geom_pos[g], 6)),
          int(m.geom_contype[g]),
          int(m.geom_conaffinity[g]),
          int(m.geom_group[g]),
        )
      )
  return out


@pytest.mark.parametrize("layout", ["apartment", "garden"])
def test_decor_leaves_the_checked_boxes_alone(layout):
  """With decor the boxes are unchanged and the added geoms touch nothing
  and are seen by no scanner (the scanners see geom group 0 only)."""
  plain, dressed = _build(layout, False), _build(layout, True)
  assert _course(plain) == _course(dressed)
  decor = [g for g in range(dressed.ngeom) if dressed.geom(g).name.startswith("decor_")]
  assert decor
  assert all(dressed.geom_contype[g] == 0 for g in decor)
  assert all(dressed.geom_conaffinity[g] == 0 for g in decor)
  assert all(dressed.geom_group[g] != 0 for g in decor)
  assert not any(plain.geom(g).name.startswith("decor_") for g in range(plain.ngeom))


@pytest.mark.parametrize("layout", ["apartment", "townhouse"])
def test_scanners_still_see_the_hidden_boxes(layout):
  """A ray cast down on each hidden box hits it, as on an undecorated one:
  MuJoCo's rays skip geoms with alpha 0, so hidden must not mean alpha 0."""
  m = _build(layout, True)
  data = mujoco.MjData(m)
  mujoco.mj_forward(m, data)
  gid = np.zeros(1, np.int32)
  group = np.array([1, 0, 0, 0, 0, 0], np.uint8)
  hidden = [
    g
    for g in range(m.ngeom)
    if m.geom(g).name.startswith("course_") and m.geom_rgba[g][3] < 0.01
  ]
  assert hidden
  for g in hidden:
    top = data.geom_xpos[g] + np.array([0.0, 0.0, m.geom_size[g][2] + 1.0])
    mujoco.mj_ray(m, data, top, np.array([0.0, 0.0, -1.0]), group, True, -1, gid)
    assert gid[0] == g, m.geom(g).name
