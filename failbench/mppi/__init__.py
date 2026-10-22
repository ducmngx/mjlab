"""MPPI recovery planner on the FailBench impact model.

Everything in this package is plain torch, with no simulator imports, so
the same planner runs in sim (``failbench/scripts/run_mppi.py``) and on the
robot: it takes the robot's own estimates (HIM state and latent, the
elevation map, the known fault, the goal in its odometry frame) and returns
the next twist command.
"""
