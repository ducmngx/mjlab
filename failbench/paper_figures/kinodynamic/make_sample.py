"""Pull one real dataset sample and the impact model's prediction -> sample.npz."""

import sys
from typing import Any

import numpy as np
import torch
from failbench.mppi.models import ImpactDynamics

P, IDX, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
L = lambda n: np.array(np.load(f"{P}/{n}.npy", mmap_mode="r")[IDX])  # noqa: E731
dyn = ImpactDynamics.from_checkpoint(
  "benchmark_results/failbench/models/impact_v5_p1.pt", "cpu", amp=False
)
T = dyn.model.cfg.num_steps
print("map_mode", dyn.map_mode, "steps", T)
m = L("map_clean")
inp = {
  "map": torch.tensor(np.stack([m, np.ones_like(m)]))[None].float(),
  "state": torch.tensor(L("state"))[None],
  "him_latent": torch.tensor(L("him_latent"))[None],
  "history": torch.tensor(L("history"))[None],
  "fault": torch.tensor(L("fault"))[None],
}
cs, cp = L("cmd_seq")[:T], L("cmd_path")[:T]
out = dyn.predict(inp, torch.tensor(cs)[None, None], torch.tensor(cp)[None, None])
o: dict[str, Any] = {k: v[0, 0].numpy() for k, v in out.items()}
print({k: v.shape for k, v in o.items()})
pose = L("pose")[:T]
print("pred end drift", o["drift"][-1], "true", pose[-1] - cp[-1])
np.savez(
  OUT,
  map=m,
  known=L("known_fused"),
  state=L("state"),
  latent=L("him_latent"),
  history=L("history"),
  fault_params=L("fault_params"),
  cmd_seq=cs,
  cmd_path=cp,
  pose=pose,
  **o,
)
