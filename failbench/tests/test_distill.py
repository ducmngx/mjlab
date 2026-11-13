"""Tests for the distilled risk critic: its map layers, the severity rule for
new entities, the student's structure, the critic built on it, the teacher's
labels and their recording."""

import numpy as np
import pytest
import torch
from failbench.distill.dump import INPUT_KEYS, LabelDump
from failbench.distill.layers import NUM_LAYERS, SEVERITY_NORM, map_layers
from failbench.distill.losses import DistillLossCfg, distill_loss, ranking_metrics
from failbench.distill.model import Student, StudentCfg
from failbench.distill.relabel import relabel_values
from failbench.mppi.cbf import severity_fields, severity_levels
from failbench.mppi.costmap import FootprintCfg, drop_cells, obstacle_cells
from failbench.mppi.distilled_critic import DistilledCritic
from failbench.mppi.risk_critic import (
  REPRESENTATIVE_FAULTS,
  RiskCritic,
  RiskCriticCfg,
  imagined_faults,
)
from failbench.severity import (
  DROP_SEVERITY,
  ENERGY_SCALE,
  SEVERITY,
  energy_scale_for,
)
from failbench.tests.test_mppi import _Thrown
from failbench.tests.test_nav2 import RES, _local_heights
from failbench.tests.test_risk_critic import _data, _walk

FP = FootprintCfg()
FAULTS = RiskCriticCfg(faults=REPRESENTATIVE_FAULTS)


def _panel(sev_value):
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  return h, torch.where(h > 0, float(sev_value), 0.0)


# The severity rule.


def test_a_known_class_keeps_its_own_energy_scale():
  for cls in ("box", "wall", "glass", "person"):
    assert energy_scale_for(SEVERITY[cls]) == pytest.approx(ENERGY_SCALE[cls])


def test_a_new_severity_takes_the_scale_between_its_neighbours():
  # Between glass (25, 5 J) and person (50, 1 J): 37.5 is halfway.
  assert energy_scale_for(37.5) == pytest.approx(3.0)
  # Outside the known range: the nearest class's.
  assert energy_scale_for(1.0) == pytest.approx(ENERGY_SCALE["wall"])
  assert energy_scale_for(80.0) == pytest.approx(ENERGY_SCALE["person"])


def test_the_teacher_prices_a_new_severity_by_the_rule():
  walk = _walk(1.0)
  thrown = _Thrown(side=-1.0)
  priced = {}
  for sev in (25.0, 37.5, 50.0):
    h, s = _panel(sev)
    priced[sev] = RiskCritic(FAULTS, thrown)(_data(h, s, walk)).item()
  assert priced[25.0] < priced[37.5] < priced[50.0]


# The map.


def test_the_map_is_terrain_known_collision_and_severity():
  h, sev = _panel(SEVERITY["glass"])
  layers = map_layers(h, sev, RES, FP)
  assert layers.shape == (1, NUM_LAYERS, *h.shape[-2:]) and NUM_LAYERS == 4
  # Something to hit: every cell the critic checks throws against, and the
  # whole of each labelled object (the critic's test marks only its edges).
  level = severity_levels(h, sev, RES, FP)
  hit = (level > 0) | (sev > 0)
  assert torch.equal(layers[:, 2], hit.float())
  assert (layers[:, 3][sev > 0] == SEVERITY["glass"] / SEVERITY_NORM).all()
  assert torch.allclose(layers[:, 3][level > 0], level[level > 0] / SEVERITY_NORM)
  assert (layers[:, 3][~hit] == 0).all()


def test_any_severity_value_reaches_the_map_unrounded():
  h, sev = _panel(37.0)
  layers = map_layers(h, sev, RES, FP)
  assert torch.allclose(layers[:, 3][sev > 0], torch.tensor(37.0 / SEVERITY_NORM))


def test_a_ledge_is_something_to_hit_at_a_ledges_severity():
  # A pit 0.6 m deep ahead of the robot.
  h = torch.zeros(1, *_local_heights([], torch.zeros(1, 3)).shape[-2:])
  w = h.shape[-1]
  h[:, :, w // 2 + 20 :] = -0.6
  layers = map_layers(h, None, RES, FP)
  drop = drop_cells(h, FP, RES)
  assert drop.any()
  assert (layers[:, 2][drop] == 1).all()
  assert (layers[:, 3][drop] == DROP_SEVERITY / SEVERITY_NORM).all()


def test_without_a_semantic_map_every_obstacle_is_a_wall():
  h, _ = _panel(0)
  layers = map_layers(h, None, RES, FP)
  obst = obstacle_cells(h, FP.wall_jump_m)
  assert (layers[:, 3][obst] == SEVERITY["wall"] / SEVERITY_NORM).all()


def test_objects_are_flattened_out_of_the_terrain_layer():
  h, sev = _panel(SEVERITY["person"])
  layers = map_layers(h, sev, RES, FP)
  assert layers[:, 0].abs().max() < 1e-6  # Flat floor, the panel removed.
  assert (layers[:, 1] == 1).all()  # Every cell known.


def test_an_object_on_a_platform_is_flattened_to_the_platform_not_the_floor():
  # A 0.2 m platform over the left half, a person standing on it.
  h = _local_heights([], torch.zeros(1, 3))
  w = h.shape[-1]
  h[..., : w // 2] = 0.2
  sev = torch.zeros_like(h)
  person = (slice(None), slice(70, 80), slice(30, 40))
  h[person] = 1.2
  sev[person] = SEVERITY["person"]
  terrain = map_layers(h, sev, RES, FP)[:, 0]
  floor = torch.median(h[h < 1.0])
  assert torch.allclose(terrain[person], 0.2 - floor, atol=1e-5)


# Relabelling.


def test_held_out_severities_are_never_drawn_for_training():
  train, held = relabel_values("train"), relabel_values("held_out")
  assert len(train) == 16 and len(held) == 15
  assert not set(train.tolist()) & set(held.tolist())
  assert train.min() >= 1.0 and train.max() <= 60.0
  # Each held-out value sits between two training values.
  assert ((held > train[:-1]) & (held < train[1:])).all()


# The student.


def _student(**kw):
  torch.manual_seed(0)
  return Student(StudentCfg(context_dim=16, **kw)).eval()


def _batch(k=3):
  h, sev = _panel(SEVERITY["glass"])
  return map_layers(h, sev, RES, FP), torch.randn(1, 16), torch.rand(1, k, 30, 3)


def test_a_faults_damage_is_each_slots_worst_step_summed_over_slots():
  student = _student()
  layers, ctx, cmd = _batch()
  out = student(layers, ctx, cmd)
  cfg = student.cfg
  assert out["per_step"].shape == (1, 3, 30, cfg.num_faults, cfg.slots)
  assert (out["per_step"] >= 0).all()
  assert torch.allclose(out["damage"], out["per_step"].amax(2).sum(-1))
  assert out["damage"].shape == (1, 3, cfg.num_faults)


def test_a_steps_damage_does_not_depend_on_later_commands():
  student = _student()
  layers, ctx, cmd = _batch(k=1)
  later = cmd.clone()
  later[:, :, 20:] = 0.0
  a = student(layers, ctx, cmd)["per_step"]
  b = student(layers, ctx, later)["per_step"]
  assert torch.allclose(a[:, :, :20], b[:, :, :20], atol=1e-5)
  assert not torch.allclose(a[:, :, 20:], b[:, :, 20:])


class _Ctx:
  """Stands in for the impact model's encoder: a fixed context vector."""

  def context(self, inputs):
    return torch.ones(len(inputs["fault"]), 16)


def test_the_distilled_critic_weighs_the_faults_as_the_teacher_does():
  student = _student()
  h, sev = _panel(SEVERITY["glass"])
  cfg = RiskCriticCfg(weight=2.0, faults=REPRESENTATIVE_FAULTS)
  critic = DistilledCritic(cfg, student, _Ctx())
  walk = _walk(torch.tensor([0.5, 1.0]).view(1, 2, 1), k=2)
  data = _data(h, sev, walk) | {"distill_layers": map_layers(h, sev, RES, FP)}
  cost = critic(data)
  assert cost.shape == (1, 2)
  _, prior = imagined_faults(cfg, 12)
  assert torch.allclose(cost, 2.0 * (critic.last["per_fault"] * prior).sum(-1))
  assert critic.__name__ == RiskCritic(cfg, _Thrown()).__name__ == "fault_risk"
  # A robot already faulted is charged nothing.
  assert (critic(data | {"exposed": torch.tensor([False])}) == 0).all()


def test_the_layers_from_already_built_levels_are_the_same():
  h, sev = _panel(SEVERITY["glass"])
  level = severity_levels(h, sev, RES, FP)
  assert torch.equal(map_layers(h, sev, RES, FP), map_layers(h, sev, RES, FP, level))
  assert len(severity_fields(h, sev, RES, FP)) == 1


# Labels and their recording.


def test_the_teacher_keeps_the_labels_the_student_learns_from():
  h, sev = _panel(SEVERITY["person"])
  critic = RiskCritic(FAULTS, _Thrown(side=-1.0))
  data = _data(h, sev, _walk(1.0, k=2)) | {"stock_cost": torch.zeros(1, 2)}
  critic(data)
  assert critic.last["per_fault"].shape == (1, 2, len(REPRESENTATIVE_FAULTS))
  assert critic.last["commands"].shape == (1, 2, 30, 3)
  assert torch.equal(critic.last["stock_cost"], data["stock_cost"])


def test_the_dump_keeps_the_cheapest_candidates_and_their_labels(tmp_path):
  n, k, f = 3, 50, 8
  cost = torch.rand(n, k)
  last = {
    "commands": torch.rand(n, k, 30, 3),
    "stock_cost": cost,
    "per_fault": torch.rand(n, k, f),
  }
  inputs = {key: torch.rand(n, 4) for key in INPUT_KEYS}
  h = torch.rand(n, 16, 16)
  sev = torch.full((n, 16, 16), 37.0)
  dump = LabelDump(tmp_path / "d", top=5, rand=10, shard_size=100)
  dump.record(last, cost, inputs, h, sev, torch.tensor([True, False, True]))
  dump.flush()
  data = np.load(dump.written[0])
  assert data["commands"].shape == (2, 15, 30, 3)
  assert data["per_fault"].shape == (2, 15, f)
  assert (data["severity"] == 37.0).all()  # Stored as is, not rounded.
  for row, robot in enumerate((0, 2)):
    cheapest = cost[robot].sort().values[:5].numpy()
    assert np.allclose(data["stock_cost"][row, :5], cheapest)
    # Each kept label belongs to its own candidate.
    i = int((cost[robot] == torch.tensor(data["stock_cost"][row, 7])).nonzero())
    assert np.allclose(data["per_fault"][row, 7], last["per_fault"][robot, i].numpy())


# Learning.


def test_a_student_matching_the_teacher_has_no_loss_and_ranks_perfectly():
  label = torch.rand(2, 40, 8) * 5
  stock = torch.rand(2, 40)
  prior = torch.full((8,), 1 / 8)
  total, _ = distill_loss(label, label, stock, prior, DistillLossCfg())
  assert total.abs() < 1e-6
  m = ranking_metrics(label, label, stock, prior, 1.0, 0.3)
  assert torch.allclose(m["spearman"], torch.ones(2))
  assert (m["top_overlap"] == 1).all() and (m["risky_abs_err"] == 0).all()
  # Reversed risk: ranks anti-correlate and the loss is positive.
  flipped = label.amax() - label
  total, _ = distill_loss(flipped, label, stock, prior, DistillLossCfg())
  assert total > 0.1
  assert (ranking_metrics(flipped, label, stock, prior, 1.0, 0.3)["spearman"] < 0).all()
