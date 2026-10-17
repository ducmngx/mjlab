"""Tests for the HIMLoco port: history reorder, estimator, actor export and
HIMPPO bookkeeping."""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from go2_him.him.algorithm import HIMPPO, NEXT_CRITIC, NEXT_VALID
from go2_him.him.env_cfg import CRITIC_SLICES, HISTORY_LENGTH, ONE_STEP_OBS_DIM
from go2_him.him.mdp import TERMINAL_CRITIC
from go2_him.him.modules import HIMActor, HIMEstimator, frame_major_index, sinkhorn
from go2_him.him.rl_cfg import go2_him_runner_cfg
from tensordict import TensorDict

TERM_DIMS = (3, 3, 3, 12, 12, 12)
NUM_ENVS = 8
ACTOR_DIM = ONE_STEP_OBS_DIM * HISTORY_LENGTH
CRITIC_DIM = CRITIC_SLICES["height"][1]


def _obs(n: int = NUM_ENVS) -> TensorDict:
  return TensorDict(
    {"actor": torch.randn(n, ACTOR_DIM), "critic": torch.randn(n, CRITIC_DIM)},
    batch_size=[n],
  )


def _runner_cfg() -> dict:
  cfg = asdict(go2_him_runner_cfg())
  cfg["multi_gpu"] = None
  cfg["num_steps_per_env"] = 4
  # Same key cleanup MjlabOnPolicyRunner does before building models.
  for key in ("actor", "critic"):
    for opt in ("cnn_cfg", "rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
      cfg[key].pop(opt, None)
  cfg["critic"].pop("distribution_cfg", None)
  return cfg


@pytest.fixture
def alg() -> HIMPPO:
  env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=12)
  out = HIMPPO.construct_algorithm(_obs(), env, _runner_cfg(), "cpu")  # type: ignore[arg-type]
  assert isinstance(out, HIMPPO)
  return out


def test_frame_major_index_reorders_term_major_history():
  # frames[k] is frame k (0 = oldest); mjlab lays it out term by term.
  frames = torch.arange(HISTORY_LENGTH * ONE_STEP_OBS_DIM, dtype=torch.float).view(
    HISTORY_LENGTH, ONE_STEP_OBS_DIM
  )
  splits = torch.split(frames, list(TERM_DIMS), dim=1)
  term_major = torch.cat([s.reshape(-1) for s in splits])
  index = frame_major_index(TERM_DIMS, HISTORY_LENGTH)
  assert torch.equal(term_major[index], frames.flip(0).reshape(-1))


def test_sinkhorn_balances_prototypes():
  # Scores are cosine similarities, as in the estimator.
  torch.manual_seed(0)
  scores = torch.rand(64, 32) * 2 - 1
  q = sinkhorn(scores)
  torch.testing.assert_close(q.sum(dim=1), torch.ones(64))
  # Columns end up far more even than a plain sharp softmax's.
  plain = torch.softmax(scores / 0.05, dim=1).sum(dim=0)
  assert q.sum(dim=0).std() < 0.5 * plain.std()


def test_estimator_learns_velocity():
  torch.manual_seed(0)
  est = HIMEstimator(history_dim=20, target_dim=10, learning_rate=1e-2)
  w = torch.randn(20, 3)
  history = torch.randn(256, 20)
  vel = history @ w
  next_obs = torch.randn(256, 10)
  first, _ = est.update(history, vel, next_obs)
  last = swap = float("nan")
  for _ in range(200):
    last, swap = est.update(history, vel, next_obs)
  assert last < 0.2 * first
  assert swap > 0.0
  v, z = est(history)
  assert v.shape == (256, 3) and z.shape == (256, 16)
  torch.testing.assert_close(z.norm(dim=1), torch.ones(256))


def test_actor_export_matches_policy(alg: HIMPPO):
  actor = alg.get_policy()
  assert isinstance(actor, HIMActor)
  obs = _obs()
  expected = actor(obs)
  jit: Any = torch.jit.script(actor.as_jit())
  torch.testing.assert_close(jit(obs["actor"]), expected)
  onnx_wrapper = cast(Any, actor.as_onnx(verbose=False))
  torch.testing.assert_close(onnx_wrapper(obs["actor"]), expected)
  assert onnx_wrapper.get_dummy_inputs()[0].shape == (1, ACTOR_DIM)


def test_estimator_is_not_in_ppo_optimizer(alg: HIMPPO):
  ppo_params = {id(p) for g in alg.optimizer.param_groups for p in g["params"]}
  assert all(id(p) not in ppo_params for p in alg.him_actor.estimator.parameters())


def test_next_obs_stored_and_terminal_steps_masked(alg: HIMPPO):
  obs = _obs()
  dones = torch.zeros(NUM_ENVS)
  dones[:3] = 1
  for _ in range(alg.storage.num_transitions_per_env):
    alg.act(obs)
    next_obs = _obs()
    alg.process_env_step(next_obs, torch.zeros(NUM_ENVS), dones, {})
    obs = next_obs
  stored = alg.storage.observations
  torch.testing.assert_close(stored[NEXT_CRITIC][-1], obs["critic"])
  assert stored[NEXT_VALID][0, :3].sum() == 0
  assert stored[NEXT_VALID][0, 3:].sum() == NUM_ENVS - 3
  alg.compute_returns(obs)
  losses = alg.update()
  assert {"estimation", "swap", "surrogate", "value"} <= losses.keys()


def test_terminal_observation_is_the_target_for_reset_envs(alg: HIMPPO):
  obs = _obs()
  alg.act(obs)
  next_obs = _obs()
  terminal = torch.randn(NUM_ENVS, CRITIC_DIM)
  dones = torch.zeros(NUM_ENVS)
  dones[:2] = 1
  extras = {TERMINAL_CRITIC: terminal}
  alg.process_env_step(next_obs, torch.zeros(NUM_ENVS), dones, extras)
  stored = alg.storage.observations
  torch.testing.assert_close(stored[NEXT_CRITIC][0, :2], terminal[:2])
  torch.testing.assert_close(stored[NEXT_CRITIC][0, 2:], next_obs["critic"][2:])
  assert stored[NEXT_VALID][0].sum() == NUM_ENVS


def test_estimator_steps_once_per_minibatch(alg: HIMPPO, monkeypatch):
  calls = []
  update = alg.him_actor.estimator.update
  monkeypatch.setattr(
    alg.him_actor.estimator,
    "update",
    lambda *a, **k: calls.append(1) or update(*a, **k),
  )
  obs = _obs()
  for _ in range(alg.storage.num_transitions_per_env):
    alg.act(obs)
    obs = _obs()
    alg.process_env_step(obs, torch.zeros(NUM_ENVS), torch.zeros(NUM_ENVS), {})
  alg.compute_returns(obs)
  alg.update()
  assert len(calls) == alg.num_learning_epochs * alg.num_mini_batches
  # The storage's own generator is restored afterwards.
  assert "mini_batch_generator" not in vars(alg.storage)


def test_save_load_round_trips_estimator(alg: HIMPPO):
  with torch.no_grad():
    for p in alg.him_actor.estimator.parameters():
      p.add_(1.0)
  saved = alg.save()
  assert "estimator_optimizer_state_dict" in saved
  env = SimpleNamespace(num_envs=NUM_ENVS, num_actions=12)
  fresh = HIMPPO.construct_algorithm(_obs(), env, _runner_cfg(), "cpu")  # type: ignore[arg-type]
  fresh.load(saved, None, strict=True)
  assert isinstance(fresh, HIMPPO)
  for a, b in zip(
    alg.him_actor.estimator.parameters(),
    fresh.him_actor.estimator.parameters(),
    strict=True,
  ):
    torch.testing.assert_close(a, b)
