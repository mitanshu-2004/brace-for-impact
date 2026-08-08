"""The bracing task: mjlab's flat walking task plus a shove the robot can see coming.

Two robots, differing in exactly one thing: whether what the robot senses includes the
warning.

  warned    senses [cos, sin, force, time_to_impact]  -> can brace before the hit
  unwarned  does not                                  -> can only respond after it

The value estimator sees the warning for both, so it is equally informed either way and
the only real difference is what the robot is allowed to act on. Everything else stays
identical: shove schedule, strengths, seed, rewards, and the ways an attempt can end.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg

from .curriculum import threat_levels
from .threat_command import ThreatCommandCfg

# The Go1 stands at 0.278 m (asset_zoo/robots/unitree_go1/go1_constants.py:81). This
# guards the degenerate solution where the policy discovers a robot lying on its belly
# cannot be knocked over; bad_orientation already covers tipping, this covers the
# level-but-grounded posture it misses.
#
# Set to 0.12, not 0.15: a passive zero-action rollout under 34 N impulses bottomed out
# at 0.151 m (smoke run, 2026-08-08). An early-training policy behaves much like that
# passive case, so a 0.15 threshold would fire constantly during the first iterations
# and truncate episodes before anything could be learned. 0.12 still sits far above a
# collapsed trunk (~0.08 m) and well below any purposeful crouch.
MIN_TRUNK_HEIGHT = 0.12


def _threat_obs() -> ObservationTermCfg:
  # Built fresh per group on purpose: sharing one cfg instance between actor and critic
  # would alias any later mutation across both groups.
  return ObservationTermCfg(
    func=mdp.generated_commands, params={"command_name": "threat"}
  )


def brace_env_cfg(
  warned: bool = True, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build the bracing env cfg.

  Args:
    warned: whether the actor observes the threat before it lands.
    play: forwarded to the underlying velocity config.
  """
  cfg = unitree_go1_flat_env_cfg(play=play)

  # The stock task applies an untelegraphed velocity kick every 1-3 s
  # (velocity_env_cfg.py:226). It is instantaneous and mass-independent -- its own
  # docstring notes it ignores inertia and contact dynamics -- so it would train
  # generic push recovery from a second, uncontrolled disturbance source and add
  # variance to both robots without telling us anything about the comparison.
  cfg.events.pop("push_robot", None)

  cfg.commands["threat"] = ThreatCommandCfg(
    # Must exceed warn_range[1] + duration_range[1] = 1.5 s so a new threat is never
    # sampled while the previous wrench is still live; ThreatCommandCfg asserts this.
    resampling_time_range=(3.0, 6.0),
    debug_vis=True,
  )

  cfg.observations["critic"].terms["threat"] = _threat_obs()
  if warned:
    cfg.observations["actor"].terms["threat"] = _threat_obs()

  # Anti-collapse guard. bad_orientation already catches tipping over; this catches
  # the level-but-grounded posture it would otherwise miss.
  cfg.terminations["collapsed"] = TerminationTermCfg(
    func=mdp.root_height_below_minimum,
    params={"minimum_height": MIN_TRUNK_HEIGHT},
  )

  cfg.curriculum["threat_levels"] = CurriculumTermCfg(
    func=threat_levels, params={"command_name": "threat"}
  )

  return cfg
