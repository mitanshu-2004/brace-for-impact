"""Throw a real ball at the robot instead of applying a wrench.

The measurements use commanded wrenches because the sweep needs an exact independent
variable. This script covers the other requirement: showing the behaviour against an
object that actually collides. The policy is the trained checkpoint, unchanged. Only
the delivery mechanism differs.

No number in the results comes from here.

`ThreatCommand` runs with `apply_wrench=False`, so it still samples the threat, counts it
down, publishes the warning to the policy and draws it, but withholds the force. The ball
is then the only thing that touches the robot, and the policy still receives exactly the
warning it was trained on. The flag changes no observation shape, so every checkpoint
stays loadable.

The ball carries `gravcomp=1` and does not fall during flight. Over a 0.6 s approach
gravity would drop a ballistic shot about 1.8 m, and correcting for that in the launch
velocity adds an error source to a shot whose only job is to arrive on time.

Mass and speed trade off against each other for a given impulse, and framing decides the
pairing. 5 kg at 6 m/s is about 30 N*s, equivalent to a 150 N wrench held 0.2 s. The same
impulse at 10 m/s would start the ball 12 m away, off camera until the final instant.

Needs an NVIDIA EGL vendor file; see docs/mjlab-notes.md.

Usage:
  MUJOCO_GL=egl python scripts/showcase.py --robot warned \
      --checkpoint logs/rsl_rl/brace_warned/model_3999.pt --out showcase.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import torch  # noqa: E402

# Shared HUD, so the showcase and the study clips are annotated identically. Works
# because `python scripts/showcase.py` puts scripts/ on sys.path[0].
from render import _font, state_row  # noqa: E402
from render import annotate as _annotate  # noqa: E402

import brace_task  # noqa: F401,E402
from brace_task import UNWARNED_TASK, WARNED_TASK  # noqa: E402

TASKS = {"warned": WARNED_TASK, "unwarned": UNWARNED_TASK}

# Somewhere far below the floor: the ball is parked here whenever it is not in flight,
# out of frame and out of contact.
PARKED = (0.0, 0.0, -20.0)


def ball_spec(radius: float, mass: float) -> mujoco.MjSpec:
  """A single free body. Follows mjlab's own cube pattern in the manipulation task."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="ball")
  body.add_freejoint(name="ball_joint")
  # Float, so a straight shot along the threat bearing lands where the wrench would
  # have. Without this the ball has to be lobbed and the aim becomes a second thing
  # that can be wrong.
  body.gravcomp = 1.0
  body.add_geom(
    name="ball_geom",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=(radius, 0.0, 0.0),
    mass=mass,
    rgba=(0.95, 0.25, 0.15, 1.0),
  )
  return spec


def annotate(frame, robot: str, ttl: float, live: bool, tilt_deg: float, speed: float,
             font, pad: int):
  return _annotate(
    frame,
    [
      (f"Unitree Go1  |  {robot}  |  {speed:.0f} m/s projectile",
       (255, 255, 255)),
      state_row(ttl, live),
      (f"trunk tilt {tilt_deg:5.1f} deg", (255, 255, 255)),
    ],
    font,
    pad,
  )


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--robot", choices=sorted(TASKS), default="warned")
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--frames", type=int, default=900)
  p.add_argument("--radius", type=float, default=0.10)
  # J = m*v*(1+e). See the module docstring for why this pairing and not a faster ball.
  p.add_argument("--mass", type=float, default=5.0)
  p.add_argument("--speed", type=float, default=6.0)
  p.add_argument("--warning-window", type=float, nargs=2, metavar=("MIN", "MAX"),
                 default=(0.6, 0.9),
                 help="warning window; narrowed from the training range so flight "
                      "distance stays in frame. Still inside what the policy saw.")
  p.add_argument("--out", default="showcase.mp4")
  p.add_argument("--width", type=int, default=1920)
  p.add_argument("--height", type=int, default=1080)
  p.add_argument("--distance", type=float, default=3.0)
  p.add_argument("--elevation", type=float, default=-10.0)
  p.add_argument("--azimuth", type=float, default=120.0)
  p.add_argument("--crf", type=int, default=16,
                 help="x264 quality, lower is better; 16 is visually lossless")
  a = p.parse_args()

  from mjlab.entity import EntityCfg
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  dev = "cuda:0"
  task = TASKS[a.robot]

  cfg = load_env_cfg(task, play=True)
  agent = load_rl_cfg(task)
  cfg.scene.num_envs = 4
  cfg.curriculum = {}

  threat = cfg.commands["threat"]
  threat.apply_wrench = False  # the ball delivers the hit, not a wrench
  threat.debug_vis = True
  threat.force_jitter = 0.0
  threat.warn_range = tuple(a.warning_window)
  cfg.commands["twist"].debug_vis = False

  cfg.scene.entities["ball"] = EntityCfg(
    spec_fn=lambda r=a.radius, m=a.mass: ball_spec(r, m),
    init_state=EntityCfg.InitialStateCfg(pos=PARKED),
  )

  cfg.viewer.width, cfg.viewer.height = a.width, a.height
  cfg.viewer.env_idx = 0
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.enable_shadows = True
  cfg.viewer.enable_reflections = True
  # ViewerConfig.origin_type defaults to AUTO, which despite its docstring resolves to a
  # free camera pinned at the world origin (see render.py). ASSET_ROOT is the branch that
  # tracks the trunk; without it the robot walks out of frame. Pose comes from the cfg
  # too, which additionally feeds _compute_render_extent's shadow/clip scaling.
  from mjlab.viewer.viewer_config import ViewerConfig

  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
  cfg.viewer.entity_name = "robot"
  cfg.viewer.distance = a.distance
  cfg.viewer.elevation = a.elevation
  cfg.viewer.azimuth = a.azimuth

  env = ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode="rgb_array")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), device=dev
  )
  runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=dev)
  policy = runner.get_inference_policy(device=dev)

  cam = env._offline_renderer._cam
  assert cam.trackbodyid >= 0, "camera is not tracking; the robot will leave frame"

  robot = env.scene["robot"]
  ball = env.scene["ball"]
  threat_term = env.command_manager.get_term("threat")
  n = env.num_envs
  parked = torch.tensor(PARKED, device=dev, dtype=torch.float32)

  def park(ids: torch.Tensor) -> None:
    state = torch.zeros(len(ids), 13, device=dev)
    state[:, :3] = parked
    state[:, 2] += torch.arange(len(ids), device=dev)  # keep parked balls apart
    state[:, 3] = 1.0  # unit quaternion
    ball.write_root_state_to_sim(state, ids)

  def launch(ids: torch.Tensor) -> None:
    """Put the ball on a straight line that reaches the trunk exactly at impact."""
    theta = threat_term._theta_w[ids]
    ttl = threat_term._time_to_impact[ids].clamp(min=1e-3)
    trunk = robot.data.root_link_pos_w[ids]

    # Lead the target. The robot walks at its commanded velocity, so over a 0.6-0.9 s
    # flight it covers 0.6-0.9 m, about its own body length and far more than the trunk
    # is wide. Aiming at the trunk's current position misses nearly every shot.
    aim = trunk + robot.data.root_link_lin_vel_w[ids] * ttl.unsqueeze(-1)

    direction = torch.stack(
      [torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1
    )
    # Travel = speed * time remaining, so it arrives as the countdown hits zero.
    start = aim - direction * (a.speed * ttl).unsqueeze(-1)
    start[:, 2] = aim[:, 2] + threat_term.cfg.impact_height

    state = torch.zeros(len(ids), 13, device=dev)
    state[:, :3] = start
    state[:, 3] = 1.0
    state[:, 7:10] = direction * a.speed
    ball.write_root_state_to_sim(state, ids)

  park(torch.arange(n, device=dev))
  in_flight = torch.zeros(n, dtype=torch.bool, device=dev)
  font = _font(max(16, a.height // 40))
  pad = max(8, a.height // 90)

  enc = dict(
    fps=50, macro_block_size=1, codec="libx264",
    output_params=["-crf", str(a.crf), "-preset", "slow", "-pix_fmt", "yuv420p"],
  )
  # Both clips stream. At 2560x1440 a 900-frame buffer is ~10 GB, and the highlights cut
  # is derived on the fly from a bounded rolling window instead of re-slicing a list.
  writer = imageio.get_writer(a.out, **enc)
  hi_out = str(Path(a.out).with_name(Path(a.out).stem + "_short.mp4"))
  hi_writer = imageio.get_writer(hi_out, **enc)

  LEAD, TAIL = 30, 60  # frames of run-up and aftermath kept around each threat
  recent: deque = deque(maxlen=LEAD)  # run-up, retained before we know a threat is due
  tail_left = 0
  n_frames = n_hi = n_windows = 0

  n_launch = n_impact = 0
  # Closest approach of ball to trunk per shot, for env 0, the only env on camera.
  # Launch and impact-window counts cannot tell a hit from a near miss: neither observes
  # the ball and the robot together. This does.
  approach: list[float] = []
  closest = float("inf")
  obs = wrapped.get_observations()
  with torch.inference_mode():
    for _ in range(a.frames):
      pending = threat_term._time_to_impact > 0
      # Launch as soon as a threat exists; flight time equals the countdown.
      fresh = pending & ~in_flight
      if fresh.any():
        launch(fresh.nonzero(as_tuple=False).flatten())
        in_flight |= fresh
        n_launch += int(fresh.sum())
      # Once the countdown has elapsed and the sustain window closed, retire the ball.
      spent = in_flight & ~pending & (threat_term._sustain_left <= 0)
      if spent.any():
        if bool(spent[0]) and closest < float("inf"):
          approach.append(closest)
          closest = float("inf")
        park(spent.nonzero(as_tuple=False).flatten())
        in_flight &= ~spent

      obs, _, _, _ = wrapped.step(policy(obs))
      if bool(in_flight[0]):
        closest = min(
          closest,
          float(
            torch.linalg.norm(
              ball.data.root_link_pos_w[0] - robot.data.root_link_pos_w[0]
            )
          ),
        )
      f = env.render()
      if f is None:
        continue
      live = bool(threat_term._sustain_left[0] > 0)
      ttl = float(threat_term._time_to_impact[0])
      n_impact += live
      tilt = torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1.0, 1.0))
      frame = annotate(
        f, a.robot, ttl, live, float(torch.rad2deg(tilt)), a.speed, font, pad
      )
      writer.append_data(frame)
      n_frames += 1

      if live or ttl > 0:
        if tail_left == 0:  # entering a fresh window: flush the retained run-up
          n_windows += 1
          for r in recent:
            hi_writer.append_data(r)
            n_hi += 1
          recent.clear()
        hi_writer.append_data(frame)
        n_hi += 1
        tail_left = TAIL
      elif tail_left > 0:
        hi_writer.append_data(frame)
        n_hi += 1
        tail_left -= 1
      else:
        recent.append(frame)

  writer.close()
  hi_writer.close()
  env.close()
  # The highlights cut comes from the same rollout rather than a second render. Threats
  # resample every 3-6 s, so most of the full clip is a robot walking. Shortening that
  # gap would move the policy off the schedule it trained on, so the full clip keeps it
  # and the cut sits alongside. Overlapping windows merge on their own: a threat arriving
  # while tail_left is still counting down extends the current window.
  print(f"wrote {n_hi} frames -> {hi_out}  ({n_windows} threat windows)")
  print(f"wrote {n_frames} frames -> {a.out}")
  print(f"  launches {n_launch}, impact-window frames {n_impact}")
  if approach:
    print("  closest approach per shot (m): "
          + ", ".join(f"{d:.2f}" for d in approach))
  assert n_launch > 0, "no projectile was ever launched; raise --frames"

  # Ball radius plus roughly half the trunk's width. A shot whose closest approach
  # exceeds this never touched the robot, and a clip of a ball sailing past is worse
  # than no clip -- it looks like the policy dodged something it never felt.
  hit_within = a.radius + 0.20
  hits = [d for d in approach if d <= hit_within]
  print(f"  {len(hits)}/{len(approach)} shots came within {hit_within:.2f} m")
  assert approach, "no shot completed; raise --frames"
  assert len(hits) >= 0.5 * len(approach), (
    f"most shots missed (closest approaches {approach}); the aim is wrong, "
    "not the policy"
  )


if __name__ == "__main__":
  main()
