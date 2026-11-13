"""The distilled fault-risk critic: ``RiskCritic``'s price from a student
network (``failbench/distill``) instead of the impact model's decoder and
the footprint sweep.

The student gives each candidate's damage per imagined fault; the faults
are weighed exactly as the teacher weighs them (``imagined_faults``'
prior, ``fault_shares``' measure, the weight), so the knob means the same.
The runner passes the map layers (``map_layers``) as
``data["distill_layers"]``; the robot's context is the impact model's
encoder, run once per robot and cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from failbench.distill.model import Student, StudentCfg
from failbench.mppi.inputs import deadband
from failbench.mppi.models import MODEL_INPUTS, ImpactDynamics
from failbench.mppi.risk_critic import RiskCriticCfg, fault_shares, imagined_faults


class ContextEncoder(Protocol):
  def context(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """``[n, context_dim]`` per robot from the impact model's inputs."""
    ...


class ImpactContext:
  """The frozen impact model's encoder memory, mean-pooled."""

  def __init__(self, model: ImpactDynamics):
    self.model = model

  @torch.no_grad()
  def context(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    dev = inputs["state"].device
    with torch.autocast("cuda", torch.bfloat16, enabled=dev.type == "cuda"):
      ctx = self.model.model.encode({k: inputs[k] for k in MODEL_INPUTS})
    return ctx.memory.float().mean(1)


def load_student(path: str | Path, device) -> Student:
  ckpt = torch.load(path, map_location=device, weights_only=False)
  student = Student(StudentCfg.from_saved(ckpt["student_cfg"])).to(device)
  student.load_state_dict(ckpt["student"])
  return student.eval()


class DistilledCritic:
  """``(data) -> cost [n, K]``, as ``RiskCritic``."""

  __name__ = "fault_risk"

  def __init__(self, cfg: RiskCriticCfg, student: Student, encoder: ContextEncoder):
    self.cfg = cfg
    self.student = student
    self.encoder = encoder
    self.last: dict[str, torch.Tensor] = {}
    """The last call's per-fault damage ``[n, K, F]``."""

  @torch.no_grad()
  def __call__(self, data: dict) -> torch.Tensor:
    c = self.cfg
    cmd = deadband(data["commands"], c.zero_below)
    n, k = cmd.shape[:2]
    _, prior = imagined_faults(c, data["inputs"]["fault"].shape[-1] // 2)
    if len(prior) != self.student.cfg.num_faults:
      raise ValueError(
        f"The student learnt {self.student.cfg.num_faults} faults; the critic "
        f"imagines {len(prior)}."
      )
    context = self.encoder.context(data["inputs"])
    feat, glob, extent = self.student.encode(data["distill_layers"])
    damage = self.student.price(feat, glob, extent, context, cmd)["damage"]
    self.last = {"per_fault": damage}
    risk = c.weight * fault_shares(damage, prior, c).sum(-1)
    exposed = data.get("exposed")
    return risk if exposed is None else risk * exposed[:, None]
