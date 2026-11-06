"""Tests for the fault-risk critic: its imagined faults, its price of one
candidate, and closed loop in Nav2 MPPI beside glass a fault would throw the
robot into."""

import torch
from failbench.faults import LOCK, NOMINAL, TORQUE, FaultMixture
from failbench.mppi.cbf import severity_fields
from failbench.mppi.costmap import FootprintCfg
from failbench.mppi.nav2 import (
  Nav2Mppi,
  Nav2MppiCfg,
  accel_limited,
  integrate,
  nav2_costmap,
)
from failbench.mppi.risk_critic import (
  RiskCritic,
  RiskCriticCfg,
  blank_obstacles,
  imagined_faults,
)
from failbench.severity import SEVERITY
from failbench.tests.test_mppi import _Thrown
from failbench.tests.test_nav2 import RES, _local_heights

HEALTHY = torch.tensor(NOMINAL).repeat(12)


def _data(h, severity, commands, n=1):
  return {
    "inputs": {"fault": HEALTHY[None].expand(n, -1)},
    "risk_fields": severity_fields(h, severity, RES, FootprintCfg()),
    "resolution": RES,
    "commands": commands,
  }


def _walk(vx, steps=30, k=1):
  cmd = torch.zeros(1, k, steps, 3)
  cmd[..., 0] = vx
  return cmd


def test_the_imagined_faults_carry_the_evaluation_mixtures_odds():
  mix = FaultMixture()
  params, prior = imagined_faults(RiskCriticCfg(), 12)
  assert torch.isclose(prior.sum(), torch.tensor(1.0))
  hurt = params != torch.tensor(NOMINAL)
  assert (hurt.any(-1).sum(-1) == 1).all()  # One joint each.
  locked = params[..., LOCK].amax(-1) > 0.5
  limp = params[..., TORQUE].amin(-1) == 0.0
  faulty = 1 - mix.probs[0]
  assert torch.isclose(prior[locked].sum(), torch.tensor(mix.probs[1] / faulty))
  limp_share = mix.probs[2] * mix.limp_prob / faulty
  assert torch.isclose(prior[limp].sum(), torch.tensor(limp_share))


def test_a_candidate_in_the_open_costs_nothing():
  h = _local_heights([], torch.zeros(1, 3))
  critic = RiskCritic(RiskCriticCfg(), _Thrown())
  data = _data(h, torch.full_like(h, 25.0), _walk(1.0))
  cost = critic(data)
  assert cost.shape == (1, 1)
  assert cost.item() == 0.0
  assert (critic.throws(_walk(1.0)[:, 0], data)["cost"] == 0).all()


def test_a_throw_costs_more_beside_worse_objects_and_at_speed():
  # A panel 0.6-0.9 m right of the line, 0.5-3 m ahead; faults throw right.
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  critic = RiskCritic(RiskCriticCfg(), _Thrown(side=-1.0))
  walk = _walk(torch.tensor([0.6, 1.0]).view(1, 2, 1), k=2)
  box = critic(_data(h, torch.full_like(h, SEVERITY["box"]), walk))
  person = critic(_data(h, torch.full_like(h, SEVERITY["person"]), walk))
  assert (box > 0).all()
  assert (person > box).all()
  assert (person[0, 1] > person[0, 0]) & (box[0, 1] > box[0, 0])  # Energy.
  # A person is hurt by any knock, a box only by a hard one: speed matters
  # less beside the person, so there keeping away does more than slowing.
  assert person[0, 1] / person[0, 0] < 0.9 * box[0, 1] / box[0, 0]


def test_the_throws_shown_are_the_ones_the_critic_prices():
  # A panel right of the line; faults on even joints throw the robot right.
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  critic = RiskCritic(RiskCriticCfg(), _Thrown(side=-1.0))
  walk = _walk(1.0)
  data = _data(h, torch.full_like(h, SEVERITY["person"]), walk)
  shown = critic.throws(walk[:, 0], data)
  params, _ = imagined_faults(RiskCriticCfg(), 12)
  assert shown["pose"].shape == (1, len(params), 30, 3)
  assert len(shown["names"]) == len(params)
  assert torch.allclose(shown["cost"].sum(-1), critic(data)[:, 0])
  # Each fault's throw: an even joint 0.8 m right by the end, as the model.
  even = shown["names"].index("FL_hip_joint:limp")
  assert torch.isclose(shown["pose"][0, even, -1, 1], torch.tensor(-0.8))


def test_the_throws_report_each_faults_touch_chance_and_damage_at_any_weight():
  # Panel on the right; side=+1: even joints throw right into it, odd left.
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  critic = RiskCritic(RiskCriticCfg(weight=0.0), _Thrown(side=1.0))
  walk = _walk(1.0)
  shown = critic.throws(walk[:, 0], _data(h, torch.full_like(h, 25.0), walk))
  right = shown["names"].index("FL_hip_joint:limp")
  left = shown["names"].index("FR_hip_joint:limp")
  assert shown["touch"].shape == shown["damage"].shape == (1, len(shown["names"]))
  assert shown["touch"][0, right] > 0.9
  assert shown["touch"][0, left] < 0.01
  # Unweighted: the expected damage given that fault, even at weight 0.
  assert 0 < shown["damage"][0, right] <= 25.0
  assert (shown["cost"] == 0).all()


def test_cvar_prices_the_worst_tail_of_the_imagined_faults():
  # A panel right of the line: only even joints' faults (thrown right) hit.
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  data = _data(h, torch.full_like(h, SEVERITY["person"]), _walk(1.0))
  mean = RiskCritic(RiskCriticCfg(), _Thrown(side=1.0))
  cvar = RiskCritic(RiskCriticCfg(measure="cvar", cvar_alpha=0.01), _Thrown(side=1.0))
  everything = RiskCritic(RiskCriticCfg(measure="cvar", cvar_alpha=1.0), _Thrown())
  _, prior = imagined_faults(RiskCriticCfg(), 12)
  worst = mean.throws(_walk(1.0)[:, 0], data)["cost"] / prior  # Per fault.
  assert torch.isclose(cvar(data)[0, 0], worst.max())  # The worst 1%: one fault.
  assert torch.isclose(everything(data)[0, 0], mean(data)[0, 0])  # All: the mean.
  shown = cvar.throws(_walk(1.0)[:, 0], data)["cost"]
  assert torch.isclose(shown.sum(), cvar(data)[0, 0])


def test_max_prices_the_single_worst_imagined_fault_whatever_its_prior():
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  data = _data(h, torch.full_like(h, SEVERITY["person"]), _walk(1.0))
  mean = RiskCritic(RiskCriticCfg(), _Thrown(side=1.0))
  # Whatever alpha says: max is not a CVaR.
  cfg = RiskCriticCfg(measure="max", cvar_alpha=1.0)
  worst = RiskCritic(cfg, _Thrown(side=1.0))
  _, prior = imagined_faults(RiskCriticCfg(), 12)
  per_fault = mean.throws(_walk(1.0)[:, 0], data)["cost"] / prior
  assert torch.isclose(worst(data)[0, 0], per_fault.max())
  shown = worst.throws(_walk(1.0)[:, 0], data)["cost"]
  assert torch.isclose(shown.sum(), worst(data)[0, 0])


def test_a_robot_already_faulted_is_charged_nothing():
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  critic = RiskCritic(RiskCriticCfg(), _Thrown(side=-1.0))
  data = _data(h, torch.full_like(h, 25.0), _walk(1.0))
  assert critic(data | {"exposed": torch.tensor([False])}).item() == 0.0


def test_obstacles_are_blanked_to_the_floor_in_the_models_map():
  height = torch.zeros(1, 8, 8)
  height[0, 2:4, 2:4] = 0.5
  model_map = torch.stack([height, torch.ones_like(height)], 1)
  out = blank_obstacles(model_map, height > 0.2)
  assert out[:, 0].abs().max() == 0.0
  assert (out[:, 1] == 1.0).all()


def test_mppi_prices_each_sample_as_it_would_be_sent():
  seen = {}

  def spy(data):
    seen.update(data)
    return data["traj"].new_zeros(data["traj"].shape[:2])

  spy.__name__ = "spy"
  cfg = Nav2MppiCfg(batch_size=64)
  planner = Nav2Mppi(cfg, 1, "cpu", extra_critics=[spy])
  h = _local_heights([], torch.zeros(1, 3))
  speed = torch.tensor([[0.5, 0.0, 0.0]])
  planner.step(nav2_costmap(h, RES, cfg.costmap), RES, torch.tensor([[5.0, 0]]), speed)
  sent = planner._constrain(seen["cv"][0], seen["v0"].expand(64, -1))
  assert torch.allclose(seen["commands"][0, :, :-1], sent[:, 1:])
  assert torch.allclose(seen["commands"][0, :, -1], sent[:, -1])


def test_the_critic_keeps_the_robot_off_the_side_a_fault_would_throw_it():
  # Glass 0.6-0.9 m right of the line, 1.5-4 m ahead; faults throw right.
  glass = [(1.5, 4.0, -0.9, -0.6)]

  def beside(weight):
    torch.manual_seed(0)
    critics = [RiskCritic(RiskCriticCfg(weight=weight), _Thrown(side=-1.0))]
    planner = Nav2Mppi(Nav2MppiCfg(batch_size=200), 1, "cpu", critics)
    cfg = planner.cfg
    pose, speed, ys = torch.zeros(1, 3), torch.zeros(1, 3), []
    for _ in range(40):
      h = _local_heights(glass, pose)
      rel = torch.tensor([[6.0, 0.0]]) - pose[:, :2]
      c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
      goal = torch.stack(
        [c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], -1
      )
      context = _data(h, torch.full_like(h, 25.0), None)
      del context["commands"]  # The planner's own.
      out = planner.step(nav2_costmap(h, RES, cfg.costmap), RES, goal, speed, context)
      cmd = out["cmd"][:, None, None].expand(-1, 1, 2, -1)
      speed = accel_limited(cmd, speed, cfg)[:, 0, 1]
      pose = integrate(speed[:, None, None], pose, cfg)[:, 0, 0]
      if 1.5 < pose[0, 0] < 4.0:
        ys.append(pose[0, 1].item())
    return sum(ys) / len(ys)

  assert beside(0.3) > beside(0.0) + 0.1


def test_with_top_m_only_the_best_samples_are_priced_and_the_rest_pay_their_worst():
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  walk = _walk(torch.linspace(0.3, 1.2, 6).view(1, 6, 1), k=6)
  data = _data(h, torch.full_like(h, 25.0), walk)
  data["stock_cost"] = torch.tensor([[5.0, 0.0, 4.0, 1.0, 3.0, 2.0]])
  full = RiskCritic(RiskCriticCfg(), _Thrown(side=-1.0))(data)
  top = RiskCritic(RiskCriticCfg(top_m=3), _Thrown(side=-1.0))(data)
  best = torch.tensor([1, 3, 5])
  assert torch.allclose(top[0, best], full[0, best])
  rest = torch.tensor([0, 2, 4])
  assert torch.allclose(top[0, rest], full[0, best].max().expand(3))
