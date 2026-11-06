"""Tests for the planner map's memory: what the LiDAR saw stays in a world
grid until a later scan sees something else there."""

import math

import torch
from failbench.sensing import MapMemory, cell_centres

CELLS, RES = 40, 0.05  # A 2 m local map.


def _scan(block_xy=None, seen_x=None) -> tuple[torch.Tensor, torch.Tensor]:
  """A local map: floor everywhere, a 1.1 m block of 0.2 m at ``block_xy``
  (local frame), ``seen_x`` limits what is known to local x >= it."""
  xy = cell_centres(CELLS, RES, "cpu")
  h = torch.zeros(1, CELLS, CELLS)
  if block_xy is not None:
    near = ((xy - torch.tensor(block_xy)).abs() < 0.1).all(-1)
    h[0, near] = 1.1
  known = torch.ones(1, CELLS, CELLS, dtype=torch.bool)
  if seen_x is not None:
    known[0] = xy[..., 0] >= seen_x
  return h, known


def _at(h, xy):
  """Height of the local cell nearest ``xy``."""
  c = cell_centres(CELLS, RES, "cpu")
  i = (c - torch.tensor(xy)).norm(dim=-1).flatten().argmin()
  return h[0].flatten()[i]


def test_a_seen_obstacle_stays_after_the_robot_walks_up_to_it_and_stops_seeing_it():
  mem = MapMemory(1, size_m=10.0, resolution=RES, device="cpu")
  z = torch.zeros(1)
  # Seen 0.6 m ahead from x = 0 ...
  h, known = _scan(block_xy=(0.6, 0.0))
  mem.update(h, known, torch.tensor([[0.0, 0.0]]), torch.zeros(1), z)
  # ... then from x = 0.5 it is beside the robot and not seen.
  h2, known2 = _scan(seen_x=0.4)
  out_h, out_known = mem.update(
    h2, known2, torch.tensor([[0.5, 0.0]]), torch.zeros(1), z
  )
  assert out_known[0].any()
  assert torch.isclose(_at(out_h, (0.1, 0.0)), torch.tensor(1.1))
  assert torch.isclose(_at(out_h, (0.6, 0.0)), torch.tensor(0.0))  # Seen now.


def test_a_later_scan_that_sees_floor_clears_a_remembered_obstacle():
  mem = MapMemory(1, size_m=10.0, resolution=RES, device="cpu")
  z, o, yaw = torch.zeros(1), torch.zeros(1, 2), torch.zeros(1)
  mem.update(*_scan(block_xy=(0.5, 0.0)), o, yaw, z)
  out_h, _ = mem.update(*_scan(), o, yaw, z)
  assert _at(out_h, (0.5, 0.0)) == 0.0


def test_memory_is_kept_in_the_world_frame_across_a_turn():
  mem = MapMemory(1, size_m=10.0, resolution=RES, device="cpu")
  z, o = torch.zeros(1), torch.zeros(1, 2)
  # Block 0.6 m ahead; the robot then turns 90 deg left and sees nothing.
  mem.update(*_scan(block_xy=(0.6, 0.0)), o, torch.zeros(1), z)
  h, known = _scan(seen_x=10.0)  # Nothing seen.
  out_h, out_known = mem.update(h, known, o, torch.full((1,), math.pi / 2), z)
  # Now on the robot's right: local (0, -0.6).
  assert torch.isclose(_at(out_h, (0.0, -0.6)), torch.tensor(1.1), atol=1e-4)
  assert _at(out_known.float(), (0.0, -0.6)) == 1.0


def test_heights_are_kept_against_the_world_so_a_trunk_bob_doesnt_move_them():
  mem = MapMemory(1, size_m=10.0, resolution=RES, device="cpu")
  o, yaw = torch.zeros(1, 2), torch.zeros(1)
  mem.update(*_scan(block_xy=(0.5, 0.0)), o, yaw, torch.zeros(1))
  h, known = _scan(seen_x=10.0)
  out_h, _ = mem.update(h, known, o, yaw, torch.full((1,), 0.1))  # Trunk 10 cm up.
  assert torch.isclose(_at(out_h, (0.5, 0.0)), torch.tensor(1.0))


def test_cells_never_seen_stay_unknown():
  mem = MapMemory(1, size_m=10.0, resolution=RES, device="cpu")
  h, known = _scan(seen_x=0.5)
  out_h, out_known = mem.update(
    h, known, torch.zeros(1, 2), torch.zeros(1), torch.zeros(1)
  )
  assert not _at(out_known.float(), (-0.5, 0.0))
  assert torch.isnan(_at(out_h, (-0.5, 0.0)))
