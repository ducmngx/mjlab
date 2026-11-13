"""Severities given to relabelled objects (``run_mppi --distill-relabel``),
so the student learns severity as a number, not four classes.

Training draws from 16 values log-spaced over 1-60; the held-out check for
new entities draws only from the 15 values between them (each the geometric
mean of its neighbours), never seen in training.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

RelabelSet = Literal["train", "held_out"]


def relabel_values(kind: RelabelSet) -> np.ndarray:
  grid = np.geomspace(1.0, 60.0, 16)
  if kind == "train":
    return grid
  return np.sqrt(grid[:-1] * grid[1:])
