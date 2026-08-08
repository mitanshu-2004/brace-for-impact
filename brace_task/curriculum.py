"""Per-environment threat difficulty curriculum.

Mirrors the shape of mjlab's ``terrain_levels_vel``: every env carries a level, an env
that survives its episode moves up, one that falls moves down. Level maps to impulse
magnitude inside :class:`ThreatCommand`, so the hits scale with what each robot can
currently take rather than with wall-clock training progress.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def threat_levels(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str = "threat",
) -> dict[str, torch.Tensor]:
  """Promote envs that survived the episode, demote those that failed."""
  threat = env.command_manager.get_term(command_name)
  levels = threat.level

  # On the very first reset nothing has been stepped yet, so no episode has been
  # survived. Promoting here would hand every env a free level -- the same trap
  # terrain_levels_vel guards against with this exact check.
  if env.common_step_counter > 0:
    # `terminated` is true only for failure terminations; time_out and
    # out_of_terrain_bounds are declared time_out=True and are excluded, so a robot
    # that simply ran out the clock counts as a survivor.
    failed = env.termination_manager.terminated[env_ids]
    delta = torch.where(failed, -1, 1).to(levels.dtype)
    levels[env_ids] = (levels[env_ids] + delta).clamp(0, threat.cfg.max_level)

  as_float = levels.float()
  return {"mean": as_float.mean(), "max": as_float.max()}
