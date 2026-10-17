"""HIMLoco runner configuration (legged_gym ``LeggedRobotCfgPPO``, runner
``HIMOnPolicyRunner``: HIMActorCritic + HIMPPO)."""

from __future__ import annotations

from dataclasses import dataclass

from go2_him.him.env_cfg import CRITIC_SLICES, HISTORY_LENGTH, NEXT_OBS_SLICE
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class HIMActorCfg(RslRlModelCfg):
  """``RslRlModelCfg`` plus ``HIMActor``'s estimator settings."""

  term_dims: tuple[int, ...] = (3, 3, 3, 12, 12, 12)
  history_length: int = HISTORY_LENGTH
  target_slice: tuple[int, int] = NEXT_OBS_SLICE
  enc_hidden_dims: tuple[int, ...] = (128, 64)
  tar_hidden_dims: tuple[int, ...] = (128, 64)
  latent_dim: int = 16
  num_prototype: int = 32
  temperature: float = 3.0
  estimator_learning_rate: float = 1e-3
  estimator_max_grad_norm: float = 10.0
  class_name: str = "go2_him.him.modules:HIMActor"


@dataclass
class HIMPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  vel_slice: tuple[int, int] = CRITIC_SLICES["lin_vel"]
  class_name: str = "go2_him.him.algorithm:HIMPPO"


def go2_him_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=HIMActorCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=False
    ),
    algorithm=HIMPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_him",
    save_interval=100,
    num_steps_per_env=100,
    max_iterations=20_000,
  )
