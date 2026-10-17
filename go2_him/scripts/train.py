"""Train the Go2 HIMLoco policy with mjlab's trainer.

  uv run python -m go2_him.scripts.train Go2-HIM-Rough --env.scene.num-envs 4096

Takes the same arguments as ``mjlab.scripts.train``; this wrapper only
registers the task first.
"""

import go2_him.him.tasks  # noqa: F401  (registers Go2-HIM-Rough)
from mjlab.scripts.train import main

if __name__ == "__main__":
  main()
