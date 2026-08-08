"""Train one robot.

mjlab's trainer does the work. This registers the two tasks and fixes the settings that
have to match across robots, so the only difference between them stays the observation
space.

Both robots use the same seed and the same PPO hyperparameters. Changing either here
changes it for both, which is the point.

Usage:
  python scripts/train.py warned --iters 4000
  python scripts/train.py unwarned     --iters 4000

Checkpoints and TensorBoard events land in logs/rsl_rl/brace_<robot>/. Watch them with:
  tensorboard --logdir logs/rsl_rl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = {
  "warned": "Mjlab-Brace-Go1-Warned",
  "unwarned": "Mjlab-Brace-Go1-Unwarned",
}


def main() -> None:
  p = argparse.ArgumentParser()
  # 4000 x 24 steps x 4096 envs = 393M environment steps, about 2.4 h on a T4.
  p.add_argument("robot", choices=sorted(ARMS))
  p.add_argument("--iters", type=int, default=4000)
  p.add_argument("--num-envs", type=int, default=4096)
  p.add_argument("--seed", type=int, default=1)
  p.add_argument("--log-root", default="logs/rsl_rl")
  a = p.parse_args()

  os.environ.setdefault("WANDB_MODE", "disabled")

  from mjlab.scripts.train import main as mjlab_train

  import brace_task  # noqa: F401  (import registers both tasks)

  sys.argv = [
    "train",
    ARMS[a.robot],
    "--env.scene.num-envs", str(a.num_envs),
    "--agent.max-iterations", str(a.iters),
    "--agent.seed", str(a.seed),
    "--agent.experiment-name", f"brace_{a.robot}",
    "--log-root", a.log_root,
  ]
  mjlab_train()


if __name__ == "__main__":
  main()
