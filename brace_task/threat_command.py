"""Telegraphed impact threats for legged locomotion.

A threat is an impulse that is *announced before it lands*. Each environment runs its
own cycle:

  resample -> countdown (threat visible in the command) -> impact (wrench written)
           -> sustain -> expire (wrench zeroed) -> idle -> resample

The command vector is what makes the task warned:

  [cos(theta_b), sin(theta_b), force / force_max, time_to_impact / warn_max]

all zeros when nothing is pending, so "no threat" is unambiguous. Dropping this vector
from the actor observation group -- and only that -- turns the task from warned
into unwarned, which is the experiment.

Implemented as a CommandTerm rather than an Event because CommandTerm already provides
observation exposure (via the stock `mdp.generated_commands` term), debug visualization,
per-episode metrics and scheduled resampling. As an Event all four would be hand-rolled,
and the observation would have to reach into the event's private state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class ThreatCommand(CommandTerm):
  """Announces, then applies, an off-centre impulse to one body."""

  cfg: ThreatCommandCfg

  def __init__(self, cfg: ThreatCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    body_ids, body_names = self.robot.find_bodies(cfg.body_name)
    if len(body_ids) != 1:
      raise ValueError(
        f"body_name={cfg.body_name!r} matched {body_names!r}; expected exactly one "
        "body to receive the impulse."
      )
    self._body_ids = body_ids

    n, dev = self.num_envs, self.device
    self._theta_w = torch.zeros(n, device=dev)  # world yaw of the applied force
    self._force = torch.zeros(n, device=dev)  # N
    self._duration = torch.zeros(n, device=dev)  # s, how long the wrench is held
    self._time_to_impact = torch.zeros(n, device=dev)  # s, > 0 while incoming
    self._sustain_left = torch.zeros(n, device=dev)  # s, > 0 while the wrench is live
    self._command_buf = torch.zeros(n, 4, device=dev)

    # Per-env difficulty. The curriculum term mutates this in place; sampling reads it.
    self.level = torch.zeros(n, dtype=torch.long, device=dev)

    self.metrics["threat_level"] = torch.zeros(n, device=dev)
    self.metrics["force_n"] = torch.zeros(n, device=dev)
    self.metrics["peak_tilt"] = torch.zeros(n, device=dev)

  @property
  def command(self) -> torch.Tensor:
    return self._command_buf

  # Sampling.

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)

    # Direction the force pushes. The threat is drawn as arriving from the opposite side.
    self._theta_w[env_ids] = r.uniform_(-math.pi, math.pi)

    # Magnitude is set by the curriculum level, jittered so each level covers a band
    # rather than a single value.
    lo, hi = self.cfg.force_range
    frac = self.level[env_ids].float() / max(self.cfg.max_level, 1)
    self._force[env_ids] = (lo + frac * (hi - lo)) * r.uniform_(
      1.0 - self.cfg.force_jitter, 1.0 + self.cfg.force_jitter
    )

    self._duration[env_ids] = r.uniform_(*self.cfg.duration_range)
    self._time_to_impact[env_ids] = r.uniform_(*self.cfg.warn_range)
    self._sustain_left[env_ids] = 0.0

  # Per-step update.

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    # env_ids is defaulted because the two mjlab lines disagree on this contract:
    # released 1.5.3 calls `self._update_command()` with no arguments once per step,
    # while main calls it with None on the per-step path and with ids on the reset
    # path (where dt is 0.0, so time must not advance). Advancing only when env_ids
    # is None is correct under both -- exactly once per step, never during a reset.
    if env_ids is None:
      self._advance(self._env.step_dt)
    self._refresh_command()

  def _advance(self, dt: float) -> None:
    incoming = self._time_to_impact > 0.0
    self._time_to_impact[incoming] -= dt

    # Impact: the countdown crossed zero on this step.
    landing = incoming & (self._time_to_impact <= 0.0)
    if landing.any():
      ids = landing.nonzero(as_tuple=False).flatten()
      # With apply_wrench=False the threat is still announced, still counted down and
      # still drawn -- only the force is withheld. The showcase uses this so a real
      # projectile can deliver the hit instead, without the robot being struck twice.
      if self.cfg.apply_wrench:
        self._write_wrench(ids)
      self._sustain_left[ids] = self._duration[ids]

    # Sustain, then expire. xfrc_applied persists until overwritten, so the wrench is
    # written once at impact and explicitly zeroed here -- never rewritten per step.
    live = self._sustain_left > 0.0
    self._sustain_left[live] -= dt
    expired = live & (self._sustain_left <= 0.0)
    if expired.any():
      self._clear_wrench(expired.nonzero(as_tuple=False).flatten())

  def _write_wrench(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    theta = self._theta_w[env_ids]
    mag = self._force[env_ids]

    forces = torch.zeros(n, 1, 3, device=self.device)
    forces[:, 0, 0] = mag * torch.cos(theta)
    forces[:, 0, 1] = mag * torch.sin(theta)

    # Apply above the centre of mass so the hit tips the robot rather than only
    # sliding it: tau = offset x F. Kept in the world frame -- "hit it high" is a
    # world-vertical notion, and it matches how the force itself is specified.
    offset = torch.zeros(n, 3, device=self.device)
    offset[:, 2] = self.cfg.impact_height
    torques = torch.cross(offset, forces[:, 0, :], dim=-1).unsqueeze(1)

    self.robot.write_external_wrench_to_sim(
      forces, torques, env_ids=env_ids, body_ids=self._body_ids
    )

  def _clear_wrench(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    zeros = torch.zeros(len(env_ids), 1, 3, device=self.device)
    self.robot.write_external_wrench_to_sim(
      zeros, zeros, env_ids=env_ids, body_ids=self._body_ids
    )
    self._sustain_left[env_ids] = 0.0

  def _refresh_command(self) -> None:
    # Bearing in the body frame: the policy needs the direction relative to itself,
    # not to the world.
    theta_b = wrap_to_pi(self._theta_w - self.robot.data.heading_w)
    pending = (self._time_to_impact > 0.0).float()
    self._command_buf[:, 0] = torch.cos(theta_b) * pending
    self._command_buf[:, 1] = torch.sin(theta_b) * pending
    # Fixed normalisers, deliberately independent of force_range and warn_range.
    # eval.py pins force_range to (F, F) to sweep one force at a time, so dividing by
    # the range maximum would report 1.0 at every level and tell the policy "maximum
    # threat" whether it faces 35 N or 250 N.
    self._command_buf[:, 2] = (self._force / self.cfg.force_norm).clamp(0.0, 1.0) * pending
    self._command_buf[:, 3] = (self._time_to_impact / self.cfg.warn_norm).clamp(0.0, 1.0)

  # Metrics. Written in place: the base reset() zeroes these tensors, so aliasing a
  # state tensor here would silently wipe the state.

  def _update_metrics(self) -> None:
    # projected_gravity_b[:, 2] is -1 when perfectly upright.
    tilt = torch.acos((-self.robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    torch.maximum(self.metrics["peak_tilt"], tilt, out=self.metrics["peak_tilt"])
    self.metrics["threat_level"].copy_(self.level.float())
    self.metrics["force_n"].copy_(self._force)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    # A live wrench persists in xfrc_applied until overwritten, so it must be cleared
    # here or it leaks into the next episode (mirrors apply_body_impulse.reset()).
    assert isinstance(env_ids, torch.Tensor)
    self._clear_wrench(env_ids)
    return super().reset(env_ids)

  # Visualization. Runs in offline rendering too, not just the live viewer:
  # render() -> update_visualizers -> command_manager.debug_vis -> here.

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    viz = self.cfg.viz
    pos_w = self.robot.data.root_link_pos_w.cpu().numpy()
    theta = self._theta_w.cpu().numpy()
    force = self._force.cpu().numpy()
    ttl = self._time_to_impact.cpu().numpy()
    live = (self._sustain_left > 0.0).cpu().numpy()

    for i in env_indices:
      anchor = pos_w[i] + np.array([0.0, 0.0, viz.z_offset])
      direction = np.array([math.cos(theta[i]), math.sin(theta[i]), 0.0])
      length = max(float(force[i]) * viz.scale, viz.min_length)

      if ttl[i] > 0.0:
        # Incoming: the arrow closes in as impact nears, so a single still frame
        # shows both that a hit is coming and how soon.
        standoff = viz.standoff + float(ttl[i]) * viz.approach_speed
        tip = anchor - direction * standoff
        visualizer.add_arrow(
          tip - direction * length, tip, color=viz.incoming_rgba, width=viz.width
        )
      elif live[i]:
        visualizer.add_arrow(
          anchor,
          anchor + direction * length,
          color=viz.impact_rgba,
          width=viz.width,
        )


@dataclass(kw_only=True)
class ThreatCommandCfg(CommandTermCfg):
  """Configuration for :class:`ThreatCommand`."""

  entity_name: str = "robot"
  body_name: str = "trunk"
  """Body that receives the impulse. Must match exactly one body."""

  force_range: tuple[float, float] = (30.0, 250.0)
  """Force at curriculum level 0 and at max_level, in Newtons. For the 13.93 kg Go1,
  F*dt/m gives dv: 35 N over 0.2 s ~ the stock push (0.5 m/s); 250 N ~ 3.6 m/s."""

  force_jitter: float = 0.15
  duration_range: tuple[float, float] = (0.1, 0.3)
  warn_range: tuple[float, float] = (0.4, 1.2)
  """Seconds of warning before impact. At step_dt = 0.02 s this is 20-60 policy steps."""

  impact_height: float = 0.10
  """Metres above the body CoM where the force acts, generating the tipping torque."""

  max_level: int = 9

  apply_wrench: bool = True
  """Whether the impulse is actually applied. False keeps the threat announced,
  counted down and drawn but withholds the force, so a real projectile can deliver the
  hit instead. Changes no observation shape, so checkpoints stay loadable."""

  # Observation scaling. Deliberately separate from force_range/warn_range so that
  # narrowing those at evaluation time does not change what the policy is told. These
  # must stay fixed across training and evaluation or the comparison is meaningless.
  force_norm: float = 250.0
  warn_norm: float = 1.2

  @dataclass
  class VizCfg:
    z_offset: float = 0.10
    scale: float = 0.004
    """Arrow length in metres per Newton."""
    min_length: float = 0.15
    width: float = 0.02
    standoff: float = 0.35
    approach_speed: float = 1.5
    """Metres of extra stand-off per second of remaining warning."""
    incoming_rgba: tuple[float, float, float, float] = (1.0, 0.15, 0.1, 0.9)
    impact_rgba: tuple[float, float, float, float] = (0.95, 0.2, 0.85, 0.95)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> ThreatCommand:
    return ThreatCommand(self, env)

  def __post_init__(self) -> None:
    # A new threat must not be sampled while the previous wrench is still live, or the
    # sustain timer of the old impulse would be clobbered and the wrench never zeroed.
    cycle = self.warn_range[1] + self.duration_range[1]
    if self.resampling_time_range[0] <= cycle:
      raise ValueError(
        f"resampling_time_range[0]={self.resampling_time_range[0]} must exceed "
        f"warn_range[1] + duration_range[1] = {cycle}, otherwise a threat can be "
        "resampled while the previous impulse is still being applied."
      )
