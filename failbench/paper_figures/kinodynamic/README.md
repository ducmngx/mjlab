# Failure-Conditioned Kinodynamic Model figure

Two layouts of the impact-model figure, drawn from one real dataset sample and
the trained model's prediction on it.

| Script | Output |
|---|---|
| `fig_column.py` | `fig_kinodynamic_col.pdf/png`, single column (3.4 in) |
| `fig_full.py` | `fig_kinodynamic.pdf/png`, wider layout (4.9 in) |

Run from the repo root:

```sh
uv run python failbench/paper_figures/kinodynamic/fig_column.py
uv run python failbench/paper_figures/kinodynamic/fig_full.py
```

Outputs go to `failbench/paper_figures/kinodynamic/out/` (or pass a directory).

## Assets (`assets/`)

Committed, so the figures render without the dataset or a GPU.

- `sample983.npz`: sample 983 of the local `course_s3_r0` probe shard (stair
  climb, rear-right thigh weakened) with `impact_v5_p1`'s prediction on it.
  Regenerate, or pick another sample:

  ```sh
  PYTHONPATH=$PWD uv run python failbench/paper_figures/kinodynamic/make_sample.py \
    <shard_dir> <index> failbench/paper_figures/kinodynamic/assets/sample983.npz
  ```

  The figure scripts load `sample983.npz` by name; change `S = np.load(...)` if
  you save a different sample. That shard was built for a probe, so the sample
  may not be held out from training; prefer a `test_terrain` sample for a
  "real prediction" claim.
- `robot_RR.png`: 3/4 Go2 render with the failed leg (rear-right) tinted red.
  `MUJOCO_GL=egl AZ=35 EL=-16 uv run python .../render_robot_side.py <out.png> RR`
  (the last argument is the leg: FL, FR, RL, RR).
- `robot_top.png`: top-down Go2 render for the plan and output tiles.
  `MUJOCO_GL=egl uv run python .../render_robot_top.py <out.png>`
