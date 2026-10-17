"""HIMPPO: PPO plus HIMLoco's estimator updates.

Port of ``rsl_rl/algorithms/him_ppo.py`` from HIMLoco (BSD-3-Clause) onto
rsl_rl 5.x's ``PPO``. Each stored transition also keeps the *next* critic
observation, which holds the estimator's targets: the next base velocity
and the next proprioception (``HIMActor.target_slice``).

As in HIMLoco:

- For a transition that ends an episode, the target is the env's terminal
  critic observation, saved before mjlab resets it
  (``mdp.store_terminal_critic`` -> ``extras[TERMINAL_CRITIC]``). Without
  that event, such transitions are left out of the estimator loss instead
  (``next_valid``).
- The estimator takes one optimizer step on each PPO mini-batch, just before
  the PPO loss of that mini-batch, at the current (adaptive) learning rate.
  It never gets PPO gradients.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from tensordict import TensorDict

from go2_him.him.mdp import TERMINAL_CRITIC
from go2_him.him.modules import VEL_DIM, HIMActor

NEXT_CRITIC = "him_next_critic"
NEXT_VALID = "him_next_valid"


class HIMPPO(PPO):
  def __init__(
    self,
    actor: HIMActor,
    critic,
    storage: RolloutStorage,
    vel_slice: tuple[int, int] | list[int] = (45, 48),
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, **kwargs)
    if not isinstance(actor, HIMActor):
      raise TypeError("HIMPPO needs an HIMActor.")
    if self.rnd is not None or self.symmetry is not None:
      raise ValueError("HIMPPO supports neither RND nor symmetry.")
    self.vel_slice = tuple(vel_slice)
    if self.vel_slice[1] - self.vel_slice[0] != VEL_DIM:
      raise ValueError(f"vel_slice must span {VEL_DIM} entries, got {vel_slice}.")
    # PPO's optimizer must not touch the estimator, which has its own.
    estimator_params = {id(p) for p in actor.estimator.parameters()}
    policy_params = [p for p in actor.parameters() if id(p) not in estimator_params]
    self.optimizer = resolve_optimizer(kwargs.get("optimizer", "adam"))(
      chain(policy_params, self.critic.parameters()), lr=self.learning_rate
    )  # type: ignore

  @property
  def him_actor(self) -> HIMActor:
    actor = self._raw_actor
    assert isinstance(actor, HIMActor)
    return actor

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
  ) -> None:
    # ``obs`` is the observation after the step: the estimator's target, except
    # for envs that just reset, whose target is their terminal observation.
    next_critic = obs["critic"]
    assert isinstance(next_critic, torch.Tensor)
    done = dones.view(-1) > 0
    valid = torch.ones_like(done)
    if bool(done.any()):
      terminal = extras.get(TERMINAL_CRITIC)
      if terminal is not None:
        next_critic = torch.where(done.unsqueeze(1), terminal, next_critic)
      else:
        valid = ~done
    current = self.transition.observations
    assert current is not None
    current = current.clone()
    current[NEXT_CRITIC] = next_critic
    current[NEXT_VALID] = valid.float().view(-1, 1)
    self.transition.observations = current
    super().process_env_step(obs, rewards, dones, extras)

  def update(self) -> dict[str, float]:
    # Run one estimator step on each mini-batch as PPO draws it, before PPO
    # computes that mini-batch's loss (HIMLoco's ordering), by wrapping the
    # storage's generator for the duration of the update.
    losses: list[tuple[float, float]] = []
    generator: Callable[..., Generator] = self.storage.mini_batch_generator

    def interleaved(num_mini_batches: int, num_epochs: int = 8):
      for batch in generator(num_mini_batches, num_epochs):
        step = self._estimator_step(batch.observations)
        if step is not None:
          losses.append(step)
        yield batch

    self.storage.mini_batch_generator = interleaved  # type: ignore[method-assign]
    try:
      loss_dict = super().update()
    finally:
      del self.storage.mini_batch_generator
    n = max(len(losses), 1)
    loss_dict["estimation"] = sum(e for e, _ in losses) / n
    loss_dict["swap"] = sum(s for _, s in losses) / n
    return loss_dict

  def _estimator_step(self, obs: TensorDict | None) -> tuple[float, float] | None:
    assert obs is not None
    actor = self.him_actor
    valid = obs[NEXT_VALID].squeeze(-1) > 0.5
    if valid.sum() < 2:
      return None
    t0, t1 = actor.target_slice
    v0, v1 = self.vel_slice
    next_critic = obs[NEXT_CRITIC]
    assert isinstance(next_critic, torch.Tensor)
    next_critic = next_critic[valid]
    return actor.estimator.update(
      actor.history(obs)[valid],
      next_critic[:, v0:v1],
      next_critic[:, t0:t1],
      learning_rate=self.learning_rate,
    )

  def save(self) -> dict:
    saved = super().save()
    saved["estimator_optimizer_state_dict"] = (
      self.him_actor.estimator.optimizer.state_dict()
    )
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    result = super().load(loaded_dict, load_cfg, strict)
    if (load_cfg is None or load_cfg.get("optimizer")) and (
      "estimator_optimizer_state_dict" in loaded_dict
    ):
      self.him_actor.estimator.optimizer.load_state_dict(
        loaded_dict["estimator_optimizer_state_dict"]
      )
    return result

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
    """As ``PPO.construct_algorithm``, with room in the rollout storage for
    the next critic observation and its validity mask."""
    alg_cfg = dict(cfg["algorithm"])
    alg_class = resolve_callable(alg_cfg.pop("class_name"))
    actor_cfg = dict(cfg["actor"])
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    critic_cfg = dict(cfg["critic"])
    critic_class = resolve_callable(critic_cfg.pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    alg_cfg.pop("rnd_cfg", None)
    alg_cfg.pop("symmetry_cfg", None)
    alg_cfg.pop("share_cnn_encoders", None)

    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg)
    actor = actor.to(device)
    print(f"Actor Model: {actor}")
    critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
    print(f"Critic Model: {critic}")

    storage_obs = obs.clone()
    storage_obs[NEXT_CRITIC] = obs["critic"]
    storage_obs[NEXT_VALID] = torch.ones(env.num_envs, 1, device=obs.device)
    storage = RolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      storage_obs,
      [env.num_actions],
      device,
    )
    alg = alg_class(
      actor, critic, storage, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"]
    )
    alg.compile(cfg.get("torch_compile_mode"))
    return alg
