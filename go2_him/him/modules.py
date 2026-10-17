"""HIMLoco's estimator and actor, ported to rsl_rl 5.x models.

Port of ``rsl_rl/modules/him_estimator.py`` and ``him_actor_critic.py`` from
github.com/InternRobotics/HIMLoco (BSD-3-Clause). Paper: Long et al.,
"Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated
Robot Response", ICLR 2024, arXiv:2312.11460.

The estimator reads a short history of proprioception and predicts the base
velocity (3) and a 16-d latent of the robot's response. The latent is
trained with a SwAV-style swapped-prediction loss against an embedding of
the next step's privileged observation, the velocity with an MSE. The actor
acts on the current observation plus both estimates; the estimator runs
without gradients there and is trained separately (see ``algorithm.py``).
"""

import copy
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.modules.distribution import Distribution
from tensordict import TensorDict

VEL_DIM = 3


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(in_dim, h), nn.ELU()]
    in_dim = h
  layers.append(nn.Linear(in_dim, out_dim))
  return nn.Sequential(*layers)


@torch.no_grad()
def sinkhorn(scores: torch.Tensor, eps: float = 0.05, iters: int = 3) -> torch.Tensor:
  """Soft, equipartitioned assignment of ``[B, K]`` scores to K prototypes
  (SwAV). Rows of the result sum to 1."""
  q = torch.exp(scores / eps).T  # [K, B]
  q /= q.sum()
  k, b = q.shape
  for _ in range(iters):
    q /= q.sum(dim=1, keepdim=True)
    q /= k
    q /= q.sum(dim=0, keepdim=True)
    q /= b
  return (q * b).T


class HIMEstimator(nn.Module):
  """Encoder from the (frame-major) observation history to velocity and
  latent, with the target network and prototypes that train the latent."""

  def __init__(
    self,
    history_dim: int,
    target_dim: int,
    enc_hidden_dims: tuple[int, ...] = (128, 64),
    tar_hidden_dims: tuple[int, ...] = (128, 64),
    latent_dim: int = 16,
    num_prototype: int = 32,
    temperature: float = 3.0,
    learning_rate: float = 1e-3,
    max_grad_norm: float = 10.0,
  ) -> None:
    super().__init__()
    self.latent_dim = latent_dim
    self.temperature = temperature
    self.max_grad_norm = max_grad_norm
    self.encoder = _mlp(history_dim, tuple(enc_hidden_dims), VEL_DIM + latent_dim)
    self.target = _mlp(target_dim, tuple(tar_hidden_dims), latent_dim)
    self.prototypes = nn.Embedding(num_prototype, latent_dim)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

  def encode(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    out = self.encoder(history)
    return out[:, :VEL_DIM], F.normalize(out[:, VEL_DIM:], dim=-1, p=2)

  def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    vel, latent = self.encode(history)
    return vel.detach(), latent.detach()

  def update(
    self,
    history: torch.Tensor,
    vel_target: torch.Tensor,
    next_obs: torch.Tensor,
    learning_rate: float | None = None,
  ) -> tuple[float, float]:
    """One optimizer step on (estimation MSE + swap loss). Inputs are rows
    with a valid next observation. Returns both losses."""
    if learning_rate is not None:
      for group in self.optimizer.param_groups:
        group["lr"] = learning_rate
    vel, z_s = self.encode(history)
    z_t = F.normalize(self.target(next_obs.detach()), dim=-1, p=2)
    with torch.no_grad():
      w = F.normalize(self.prototypes.weight.data.clone(), dim=-1, p=2)
      self.prototypes.weight.copy_(w)
    score_s = z_s @ self.prototypes.weight.T
    score_t = z_t @ self.prototypes.weight.T
    q_s = sinkhorn(score_s)
    q_t = sinkhorn(score_t)
    log_p_s = F.log_softmax(score_s / self.temperature, dim=1)
    log_p_t = F.log_softmax(score_t / self.temperature, dim=1)
    swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
    estimation_loss = F.mse_loss(vel, vel_target.detach())
    loss = estimation_loss + swap_loss
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    self.optimizer.step()
    return estimation_loss.item(), swap_loss.item()


def frame_major_index(term_dims: tuple[int, ...], history_length: int) -> torch.Tensor:
  """Index that reorders mjlab's term-major history into frames, newest
  first (HIMLoco's ``obs_buf`` layout).

  mjlab flattens a group with history as ``[term_0(t-H+1..t), term_1(...),
  ...]``, each term's frames oldest first. The result is ``[frame t,
  frame t-1, ...]``, each frame ``[term_0, term_1, ...]``; so the first
  ``sum(term_dims)`` entries are the current observation.
  """
  offsets = [0]
  for d in term_dims[:-1]:
    offsets.append(offsets[-1] + d * history_length)
  index = []
  for k in reversed(range(history_length)):
    for off, d in zip(offsets, term_dims, strict=True):
      start = off + k * d
      index.extend(range(start, start + d))
  return torch.tensor(index, dtype=torch.long)


class HIMActor(MLPModel):
  """HIMLoco actor: MLP on [current obs, estimated velocity, latent].

  Reads the ``actor`` observation set, which must be the one-step terms
  ``term_dims`` with ``history_length`` frames (see ``env_cfg.py``).
  ``target_slice`` picks the estimator's contrastive target from the
  critic observation.
  """

  frame_index: torch.Tensor
  estimator: HIMEstimator

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    term_dims: tuple[int, ...] | list[int] = (3, 3, 3, 12, 12, 12),
    history_length: int = 6,
    target_slice: tuple[int, int] | list[int] = (3, 48),
    enc_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    tar_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    latent_dim: int = 16,
    num_prototype: int = 32,
    temperature: float = 3.0,
    estimator_learning_rate: float = 1e-3,
    estimator_max_grad_norm: float = 10.0,
  ) -> None:
    if obs_normalization:
      raise ValueError("HIMActor follows HIMLoco and does not normalize inputs.")
    self.one_step_dim = int(sum(term_dims))
    self.latent_dim = latent_dim
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
    )
    history_dim = self.one_step_dim * history_length
    if self.obs_dim != history_dim:
      raise ValueError(
        f"Actor observation is {self.obs_dim}-d; expected {history_length} frames "
        f"of {self.one_step_dim} ({history_dim})."
      )
    self.register_buffer(
      "frame_index", frame_major_index(tuple(term_dims), history_length)
    )
    self.target_slice = tuple(target_slice)
    self.estimator = HIMEstimator(
      history_dim,
      self.target_slice[1] - self.target_slice[0],
      enc_hidden_dims=tuple(enc_hidden_dims),
      tar_hidden_dims=tuple(tar_hidden_dims),
      latent_dim=latent_dim,
      num_prototype=num_prototype,
      temperature=temperature,
      learning_rate=estimator_learning_rate,
      max_grad_norm=estimator_max_grad_norm,
    )

  def _get_latent_dim(self) -> int:
    return self.one_step_dim + VEL_DIM + self.latent_dim

  def history(self, obs: TensorDict) -> torch.Tensor:
    """Frame-major, newest-first observation history ``[B, H * one_step]``."""
    parts = [cast(torch.Tensor, obs[g]) for g in self.obs_groups]
    return torch.cat(parts, dim=-1)[:, self.frame_index]

  def estimate(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    """The estimator's velocity (in the critic's scaled units, i.e. x2 m/s)
    and latent for the current step."""
    return self.estimator(self.history(obs))

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    hist = self.history(obs)
    with torch.no_grad():
      vel, latent = self.estimator(hist)
    return torch.cat([hist[:, : self.one_step_dim], vel, latent], dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    pass

  def as_jit(self) -> nn.Module:
    return _ExportHIMActor(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _OnnxHIMActor(self, verbose)


class _ExportHIMActor(nn.Module):
  """Deterministic HIM policy on the flat ``actor`` observation (mjlab's
  term-major history), for TorchScript."""

  frame_index: torch.Tensor

  def __init__(self, model: HIMActor) -> None:
    super().__init__()
    self.register_buffer("frame_index", model.frame_index.clone())
    self.one_step_dim = model.one_step_dim
    self.vel_dim = VEL_DIM
    self.encoder = copy.deepcopy(model.estimator.encoder)
    self.mlp = copy.deepcopy(model.mlp)
    distribution = model.distribution
    assert isinstance(distribution, Distribution)
    self.deterministic_output = distribution.as_deterministic_output_module()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    hist = x[:, self.frame_index]
    out = self.encoder(hist)
    vel = out[:, : self.vel_dim]
    latent = F.normalize(out[:, self.vel_dim :], dim=-1, p=2.0)
    h = torch.cat([hist[:, : self.one_step_dim], vel, latent], dim=-1)
    return self.deterministic_output(self.mlp(h))

  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxHIMActor(_ExportHIMActor):
  is_recurrent: bool = False

  def __init__(self, model: HIMActor, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.input_size = model.obs_dim

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
