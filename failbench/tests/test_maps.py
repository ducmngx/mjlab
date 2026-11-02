"""Tests for the phantom maps and the global planner that plans paths on
them (the planner-driven data's global paths)."""

import numpy as np
import torch
from failbench.global_planner import plan_paths
from failbench.maps import LAYOUTS, MapCfg, map_costmap, rasterize, sample_map
from failbench.mppi.nav2 import INSCRIBED, _lookup


def test_a_box_fills_the_cells_it_covers_row_y_column_x():
  # A 1 m x 0.4 m box centred 2 m along +x, 1 m along -y, unrotated.
  boxes = torch.tensor([[[2.0, -1.0, 0.5, 0.2, 0.0]]])
  occ = rasterize(boxes, cells=120, resolution=0.1)[0]
  rows, cols = occ.nonzero(as_tuple=True)
  x = (cols.float() - 59.5) * 0.1
  y = (rows.float() - 59.5) * 0.1
  assert abs(x.mean() - 2.0) < 0.06 and abs(y.mean() + 1.0) < 0.06
  assert 1.0 - 0.15 < x.max() - x.min() + 0.1 < 1.0 + 0.15
  assert 0.4 - 0.15 < y.max() - y.min() + 0.1 < 0.4 + 0.15


def test_every_layout_stays_on_its_map():
  cfg = MapCfg()
  rng = np.random.default_rng(0)
  half = cfg.size_m / 2
  for kind in LAYOUTS:
    for _ in range(20):
      boxes = sample_map(kind, rng, cfg)
      used = boxes[:, 2] > 0
      assert used.any() or kind == "open", kind
      assert (np.abs(boxes[used, :2]) <= half).all(), kind


def _wall_with_a_gap():
  """A wall across the map at x = 0 with a 1.2 m door at y = 2."""
  wall = [
    [0.0, -2.3, 0.1, 3.7, 0.0],  # y -6 .. 1.4
    [0.0, 4.3, 0.1, 1.7, 0.0],  # y 2.6 .. 6
  ]
  return torch.tensor([wall])


def test_a_planned_path_goes_through_the_door_clear_of_the_wall():
  cfg = MapCfg()
  boxes = _wall_with_a_gap()
  cost = map_costmap(boxes, cfg)[0]
  start, goal = np.array([[-3.0, -3.0]]), np.array([[3.0, -3.0]])
  paths = plan_paths(cost.numpy(), cfg.resolution, start, goal)
  assert len(paths) == 1 and paths[0] is not None
  path = paths[0]
  assert np.linalg.norm(path[0] - start[0]) < 0.1
  assert np.linalg.norm(path[-1] - goal[0]) < 0.1
  # Through the door (y about 2), far longer than the straight 6 m.
  crossing = path[np.argmin(np.abs(path[:, 0]))]
  assert 1.4 < crossing[1] < 2.6
  assert np.linalg.norm(np.diff(path, axis=0), axis=-1).sum() > 10.0
  # Never where the robot's body would touch the wall.
  c = _lookup(cost[None], torch.tensor(path, dtype=torch.float32)[None], cfg.resolution)
  assert (c < INSCRIBED).all()


def test_no_path_where_the_goal_is_walled_off():
  cfg = MapCfg()
  boxes = torch.tensor([[[0.0, 0.0, 0.1, 6.0, 0.0]]])  # A wall all the way.
  cost = map_costmap(boxes, cfg)[0]
  paths = plan_paths(
    cost.numpy(), cfg.resolution, np.array([[-3.0, 0.0]]), np.array([[3.0, 0.0]])
  )
  assert paths == [None]


def test_a_path_across_open_ground_is_straight():
  cfg = MapCfg()
  cost = map_costmap(torch.zeros(1, 1, 5), cfg)[0]
  start, goal = np.array([[-4.0, -4.0]]), np.array([[4.0, 3.0]])
  path = plan_paths(cost.numpy(), cfg.resolution, start, goal)[0]
  assert path is not None
  d = goal[0] - start[0]
  normal = np.array([-d[1], d[0]]) / np.linalg.norm(d)
  assert np.abs((path - start[0]) @ normal).max() < 0.05
