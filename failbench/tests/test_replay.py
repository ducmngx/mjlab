"""Pricing a replayed fault's touches against a layout's objects."""

from __future__ import annotations

import math

import torch
from failbench.house import Item, Layout
from failbench.replay import ObjectBoxes, price_touches
from failbench.severity import ENERGY_SCALE, MASS_KG, SEVERITY

LAY = Layout(
  name="t",
  size=(4.0, 4.0),
  items=(
    Item("floor", 0.0, 4.0, 0.0, 4.0, -1.0, 0.0),
    Item("glass", 1.0, 1.5, 1.0, 1.5, 0.0, 1.0, "case"),
    Item("person", 3.0, 3.4, 1.0, 1.3, 0.0, 1.7, "p"),
    Item("box", 1.0, 2.0, 3.0, 3.5, 0.0, 0.8, "sofa"),
  ),
  places={},
  routes=(),
)


def _price(points, touch, speed):
  boxes = ObjectBoxes.from_layout(LAY, "cpu")
  pts = torch.tensor(points, dtype=torch.float32)[None]
  return price_touches(boxes, pts, torch.tensor([touch]), torch.tensor([speed**2]))


def test_a_touch_on_glass_costs_its_damage_at_the_trunks_speed():
  sev, dmg = _price([(1.51, 1.2, 0.3)], [True], 1.0)  # 1 cm off its face.
  energy = 0.5 * MASS_KG
  want = SEVERITY["glass"] * (1 - math.exp(-energy / ENERGY_SCALE["glass"]))
  assert sev.item() == SEVERITY["glass"]
  assert abs(dmg.item() - want) < 1e-4


def test_floor_and_far_contacts_cost_nothing_and_untouched_ones_are_ignored():
  # A knee on the floor, a point 20 cm from the glass, and a contact on the
  # person that isn't a touch (a foot slot, say).
  sev, dmg = _price(
    [(2.5, 2.5, 0.0), (1.7, 1.2, 0.3), (3.2, 1.15, 1.0)], [True, True, False], 1.0
  )
  assert sev.item() == 0.0 and dmg.item() == 0.0


def test_the_worst_object_touched_sets_the_cost():
  sev, dmg = _price([(1.95, 3.25, 0.4), (3.2, 0.99, 1.0)], [True, True], 0.5)
  assert sev.item() == SEVERITY["person"]
  energy = 0.5 * MASS_KG * 0.25
  want = SEVERITY["person"] * (1 - math.exp(-energy / ENERGY_SCALE["person"]))
  assert abs(dmg.item() - want) < 1e-4


def test_distance_to_objects_is_from_the_nearest_point_to_each_box_surface():
  from failbench.replay import object_distance

  boxes = ObjectBoxes.from_layout(LAY, "cpu")
  # Two body points: one 0.3 m right of the glass case, one 0.1 m below the
  # sofa's near face.
  pts = torch.tensor([[(1.8, 1.25, 0.3), (1.5, 2.9, 0.3)]])
  d = object_distance(boxes, pts)  # [B, I]
  glass, person, sofa = d[0].tolist()
  assert abs(glass - 0.3) < 1e-5
  assert abs(sofa - 0.1) < 1e-5
  assert abs(person - 1.2) < 1e-5  # Level with it: straight across in x.


def test_the_peak_force_counts_only_contacts_on_objects():
  from failbench.replay import object_contacts

  boxes = ObjectBoxes.from_layout(LAY, "cpu")
  pts = torch.tensor([[(1.51, 1.2, 0.3), (2.5, 2.5, 0.0), (3.2, 0.99, 1.0)]])
  touch = torch.tensor([[True, True, True]])
  hit = object_contacts(boxes, pts, touch)  # [B, N, I]
  force = torch.tensor([[40.0, 300.0, 15.0]])
  on_obj = torch.where(hit.any(-1), force, 0.0).amax(-1)
  assert on_obj.item() == 40.0  # The floor contact's 300 N doesn't count.
  person = boxes.severity == 50.0
  on_person = torch.where((hit & person).any(-1), force, 0.0).amax(-1)
  assert on_person.item() == 15.0


def test_the_fault_window_records_collisions_forces_and_distances_in_its_horizon():
  from failbench.replay import FaultWindow

  boxes = ObjectBoxes.from_layout(LAY, "cpu")
  win = FaultWindow(boxes, window_s=2.0, n=2, device="cpu")
  nan = float("nan")
  far = torch.tensor([[(2.5, 2.5, 0.3)], [(2.5, 2.5, 0.3)]])  # 1 m from the glass.
  near = torch.tensor([[(3.2, 0.7, 0.3)], [(2.5, 2.5, 0.3)]])  # 0.3 m from the person.
  none = torch.zeros(2, 1, 3)
  no_touch = torch.zeros(2, 1, dtype=torch.bool)
  zero = torch.zeros(2, 1)
  # Before env 0's fault (env 1 never faults): a person touch doesn't count.
  on_person = torch.tensor([[(3.2, 0.99, 1.0)], [(0.0, 0.0, 0.0)]])
  touch0 = torch.tensor([[True], [False]])
  win.update(
    torch.tensor([nan, nan]),
    near,
    on_person,
    touch0,
    zero + 99,
    torch.ones(2),
    torch.zeros(2, dtype=torch.bool),
  )
  for since, pts, cpts, touch, force in (
    (0.5, far, none, no_touch, zero),
    (1.0, near, on_person, touch0, torch.tensor([[60.0], [0.0]])),
    (2.0, near, none, no_touch, zero),
    (2.5, far, on_person, touch0, zero + 500),  # After the window.
  ):
    win.update(
      torch.tensor([since, nan]),
      pts,
      cpts,
      touch,
      force,
      torch.ones(2),
      torch.zeros(2, dtype=torch.bool),
    )
  r = win.results()
  assert r["collided"][0] and r["collided_person"][0] and not r["collided_glass"][0]
  assert r["peak_force_n"][0] == 60.0 and r["peak_force_person_n"][0] == 60.0
  assert abs(r["impulse_ns"][0] - 60.0 * 0.02) < 1e-6  # One 0.02 s step.
  assert r["first_contact_s"][0] == 1.0
  assert abs(r["min_dist_person_m"][0] - 0.3) < 1e-5
  assert abs(r["end_dist_person_m"][0] - 0.3) < 1e-5  # At 2.0 s, not after.
  assert r["complete"][0]
  # Env 1 never faulted: no window at all.
  assert not r["complete"][1] and not r["collided"][1]
  assert math.isnan(r["end_dist_m"][1])


def test_a_foot_on_the_floor_beside_an_object_doesnt_touch_it():
  from failbench.replay import object_contacts

  boxes = ObjectBoxes.from_layout(LAY, "cpu")
  # A foot planted 1 cm from the sofa's face, and one kicking it 10 cm up.
  pts = torch.tensor([[(1.5, 2.99, 0.0), (1.5, 2.99, 0.1)]])
  hit = object_contacts(boxes, pts, torch.tensor([[True, True]]))
  assert hit.any(-1)[0].tolist() == [False, True]
