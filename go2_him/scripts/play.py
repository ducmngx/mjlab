"""Play a trained Go2 HIMLoco policy in mjlab's viewer.

  uv run python -m go2_him.scripts.play Go2-HIM-Rough --checkpoint-file <model.pt>

Takes the same arguments as ``mjlab.scripts.play``; this wrapper only
registers the task first.
"""

import go2_him.him.tasks  # noqa: F401  (registers Go2-HIM-Rough)
from mjlab.scripts.play import main

if __name__ == "__main__":
  main()
