"""Registers the two bracing tasks.

Importing this package is what makes the task ids resolvable, exactly as mjlab's own
`tasks/velocity/config/go1/__init__.py` does for the stock tasks.
"""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .brace_env_cfg import brace_env_cfg
from .curriculum import threat_levels
from .threat_command import ThreatCommand, ThreatCommandCfg

__all__ = [
  "ThreatCommand",
  "ThreatCommandCfg",
  "brace_env_cfg",
  "threat_levels",
  "WARNED_TASK",
  "UNWARNED_TASK",
]

WARNED_TASK = "Mjlab-Brace-Go1-Warned"
UNWARNED_TASK = "Mjlab-Brace-Go1-Unwarned"

# The PPO hyperparameters are the stock Go1 ones, unchanged. The comparison is between
# two observation spaces, so retuning either robot would muddy it.
register_mjlab_task(
  task_id=WARNED_TASK,
  env_cfg=brace_env_cfg(warned=True),
  play_env_cfg=brace_env_cfg(warned=True, play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id=UNWARNED_TASK,
  env_cfg=brace_env_cfg(warned=False),
  play_env_cfg=brace_env_cfg(warned=False, play=True),
  rl_cfg=unitree_go1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
