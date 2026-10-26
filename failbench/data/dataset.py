"""PyTorch dataset over built FailBench sample shards (``build.py``).

Shards are memory-mapped. Each item is a dict of tensors:

- ``map`` ``[2, H, W]``: LiDAR-realistic height (0 where unknown) and the
  known mask (``map_mode``: ``fused`` / ``single`` visibility with
  ``corrupt_map`` noise, or ``clean``: every cell, no noise);
- ``state`` (33), ``state_true_vel`` (3), ``him_latent`` (16), ``history``
  ``[20, 52]`` (``[20, 36]`` before the latent), ``cmd_seq`` ``[T, 3]``, ``fault`` (24);
- targets ``dv`` / ``pose`` / ``cmd_path`` ``[T, 3]``, ``upright`` / ``collision`` /
  ``valid`` / ``contact_offmap`` ``[T]``, ``safe`` ``[T]`` and
  ``latent_future`` ``[T, 16]`` (shards built with them), and with
  ``contacts`` ``contact``
  ``[T, H, W]`` rasterised from the CSR rows;
- labels ``fault_family``, ``plan_hold``, ``terrain_id``, ``offset_s``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from failbench.data.build import SPLITS
from failbench.sensing import NoiseCfg, corrupt_map

MapMode = Literal["fused", "single", "clean"]

_PER_SAMPLE = (
  "state",
  "state_true_vel",
  "him_latent",
  "history",
  "cmd_seq",
  "fault",
  "dv",
  "pose",
  "cmd_path",
  "upright",
  "collision",
  "valid",
  "contact_offmap",
  "fault_family",
  "plan_hold",
  "offset_s",
)


_OPTIONAL = ("safe", "latent_future")
"""Targets only shards built since they were added have."""


class FailBenchDataset(Dataset):
  def __init__(
    self,
    root: str | Path,
    split: str = "train",
    map_mode: MapMode = "fused",
    noise: NoiseCfg | None = None,
    drop_fallen_at_start: bool = True,
    contacts: bool = True,
    seed: int | None = None,
  ):
    """``noise`` defaults to ``NoiseCfg()``. ``seed`` fixes each item's map
    noise (for val/test); ``None`` draws fresh noise every time. ``contacts``
    rasterises the contact maps (only a model with contact heads needs
    them)."""
    self.contacts = contacts
    self.map_mode = map_mode
    self.noise = noise or NoiseCfg()
    self.seed = seed
    code = SPLITS.index(split)
    self.shards: list[dict[str, np.ndarray]] = []
    self.metas: list[dict] = []
    index = []
    dirs = sorted(p.parent for p in Path(root).glob("*/meta.json"))
    if not dirs:
      raise FileNotFoundError(f"No built shards under {root}.")
    for s, d in enumerate(dirs):
      arrays = {p.stem: np.load(p, mmap_mode="r") for p in d.glob("*.npy")}
      sel = arrays["split"] == code
      if drop_fallen_at_start:
        sel &= ~arrays["fallen_at_start"]
      rows = np.flatnonzero(sel)
      index.append(np.stack([np.full_like(rows, s), rows], axis=1))
      self.shards.append(arrays)
      self.metas.append(json.loads((d / "meta.json").read_text()))
    self.index = np.concatenate(index)
    meta = self.metas[0]
    self.num_bins: int = meta["num_bins"]
    self.cells: int = meta["map_cells"]

  def __len__(self) -> int:
    return len(self.index)

  def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
    s, row = (int(x) for x in self.index[index])
    a = self.shards[s]
    keys = (*_PER_SAMPLE, *(k for k in _OPTIONAL if k in a))
    item = {k: torch.from_numpy(np.array(a[k][row])) for k in keys}
    item["terrain_id"] = torch.tensor(self.metas[s]["terrain_id"])

    height = torch.from_numpy(a["map_clean"][row].astype(np.float32))
    if self.map_mode == "clean":
      known = ~torch.isnan(height)
      item["map"] = torch.stack([torch.nan_to_num(height), known.float()])
    else:
      known = torch.from_numpy(np.array(a[f"known_{self.map_mode}"][row]))
      gen = None
      if self.seed is not None:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + index)
      item["map"] = corrupt_map(height, known, self.noise, gen)

    if not self.contacts:
      return item
    lo, hi = int(a["contact_ptr"][row]), int(a["contact_ptr"][row + 1])
    contact = torch.zeros(self.num_bins, self.cells, self.cells, dtype=torch.bool)
    if hi > lo:
      b = torch.from_numpy(a["contact_bin"][lo:hi].astype(np.int64))
      iy = torch.from_numpy(a["contact_iy"][lo:hi].astype(np.int64))
      ix = torch.from_numpy(a["contact_ix"][lo:hi].astype(np.int64))
      contact[b, iy, ix] = True
    item["contact"] = contact
    return item
