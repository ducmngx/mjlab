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
  cost = critic(_data(h, torch.full_like(h, 25.0), _walk(1.0)))
  assert cost.shape == (1, 1)
  assert cost.item() == 0.0


def test_a_throw_costs_more_beside_worse_objects_and_at_speed():
  # A panel 0.6-0.9 m right of the line, 0.5-3 m ahead; faults throw right.
  h = _local_heights([(0.5, 3.0, -0.9, -0.6)], torch.zeros(1, 3))
  critic = RiskCritic(RiskCriticCfg(), _Thrown(side=-1.0))
  walk = _walk(torch.tensor([0.6, 1.0]).view(1, 2, 1), k=2)
  box = critic(_data(h, torch.full_like(h, 5.0), walk))
  glass = critic(_data(h, torch.full_like(h, 25.0), walk))
  assert (box > 0).all()
  assert torch.allclose(glass, 5 * box)
  assert glass[0, 1] > glass[0, 0]  # Kinetic energy at the hit.


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

  assert beside(0.01) > beside(0.0) + 0.1


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
