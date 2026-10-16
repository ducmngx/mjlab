"""Tests for the obstacle-lane course terrain."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
from heat_bench.envs_mjlab.course_terrain import (
  Floor,
  FloorPiece,
  floor_profile,
  profile_height,
)
from heat_bench.envs_mjlab.go2_eval_env_cfg import COURSE_TERRAINS_CFG

from mjlab.terrains.terrain_generator import TerrainGenerator


@pytest.fixture(scope="module")
def course():
  """The course preset with three difficulty rows (0, 0.5, 1), compiled."""
  gen = TerrainGenerator(replace(COURSE_TERRAINS_CFG, num_rows=3))
  spec = mujoco.MjSpec()
  gen.compile(spec)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return gen, model, data


def _ray_down(model, data, x: float, y: float) -> tuple[float, int]:
  geom = np.zeros(1, np.int32)
  dist = mujoco.mj_ray(
    model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]), None, True, -1, geom
  )
  return 5.0 - dist, int(geom[0])


def test_geom_names_unique(course):
  _, model, _ = course
  names = [model.geom(i).name for i in range(model.ngeom)]
  assert len(set(names)) == len(names)


def test_lane_is_clear_floor_with_no_gaps(course):
  """Rays down the lane (+-0.5 m) only ever hit floor geoms."""
  gen, model, data = course
  for row in range(gen.cfg.num_rows):
    for col in range(gen.terrain_origins.shape[1]):
      ox, oy, _ = gen.terrain_origins[row, col]
      for x in np.arange(-0.8, 14.8, 0.1):
        for dy in (-0.5, 0.0, 0.5):
          _, geom = _ray_down(model, data, ox + x, oy + dy)
          assert geom >= 0, (col, row, x, dy)
          assert model.geom(geom).name.startswith("course_floor"), (col, row)


def test_obstacles_stay_outside_the_lane(course):
  gen, model, data = course
  num_obstacles = 0
  for i in range(model.ngeom):
    name = model.geom(i).name
    if not name.startswith("course_") or name.startswith("course_floor"):
      continue
    num_obstacles += 1
    pos = data.geom_xpos[i]
    tiles = gen.terrain_origins.reshape(-1, 3)
    lane_y = tiles[np.argmin(np.linalg.norm(tiles[:, :2] - pos[:2], axis=1)), 1]
    size = model.geom_size[i]
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
      rot = data.geom_xmat[i].reshape(3, 3)
      half_y = abs(rot[1, 0]) * size[0] + abs(rot[1, 1]) * size[1]
    else:
      half_y = size[0]
    assert abs(pos[1] - lane_y) - half_y >= 0.55 - 1e-6, name
  assert num_obstacles > 0


def test_spawn_is_on_flat_floor(course):
  gen, model, data = course
  for row in range(gen.cfg.num_rows):
    for col in range(gen.terrain_origins.shape[1]):
      ox, oy, oz = gen.terrain_origins[row, col]
      for dx in (-0.3, 0.0, 0.3):
        z, _ = _ray_down(model, data, ox + dx, oy)
        assert z == pytest.approx(oz, abs=1e-3), (col, row)


def _profile(floor: Floor, difficulty: float) -> list[FloorPiece]:
  return floor_profile(
    floor,
    difficulty,
    length=16.0,
    feature_start=3.0,
    slope_deg_range=(5.0, 20.0),
    ramp_length=3.0,
    riser_range=(0.03, 0.14),
    tread=0.3,
    num_steps_range=(5, 8),
  )


@pytest.mark.parametrize(
  ("up", "down"), [("slope_up", "slope_down"), ("stairs_up", "stairs_down")]
)
def test_features_grow_with_difficulty(up: Floor, down: Floor):
  top = [profile_height(_profile(up, d), np.array([15.0]))[0] for d in (0.0, 0.5, 1.0)]
  assert top[0] > 0.0
  assert top[0] < top[1] < top[2]
  # Down variants mirror the profile: start at the top, end at zero.
  pieces = _profile(down, 1.0)
  assert pieces[0].z0 == pytest.approx(top[2])
  assert pieces[-1].z1 == pytest.approx(0.0, abs=1e-9)
