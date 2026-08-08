"""Render a single robot close-up with the threat drawn.

A red arrow closes in on the robot, the stance widens, then the impulse lands. Two
properties of mjlab's offscreen renderer shape how the camera is set up here.

Debug arrows are drawn for one environment only. `offscreen_renderer.update()`
constructs the visualizer without `show_all_envs`, so only `viewer.env_idx` is
annotated. `max_extra_envs = 0` leans into that and renders a single robot, which reads
better in close-up than a crowd with one annotated member.

The camera has to be asked to track. `ViewerConfig.origin_type` defaults to AUTO, and
`_setup_camera()` handles AUTO in the same branch as WORLD: `mjCAMERA_FREE` with
`trackbodyid = -1`, a fixed camera at the world origin. A policy walking at its
commanded velocity leaves the frame in a few seconds. ASSET_ROOT with `entity_name` is
the branch that tracks a body.

Camera pose comes from the cfg rather than from `renderer._cam` after construction,
because `_compute_render_extent()` derives the render extent from `cfg.distance`, and
MuJoCo scales z-near, z-far, and the shadow clip off `model.stat.extent`. Setting the
pose anywhere else leaves the extent computed for a distance the camera is not at.

Needs an NVIDIA EGL vendor file. Kaggle images ship only Mesa and cannot get a context.

Usage:
  MUJOCO_GL=egl python scripts/render.py --robot warned \
      --checkpoint model_3999.pt --force 160 --out brace_160N.mp4

  # training progression: one clip sweeping checkpoints, labelled by iteration
  MUJOCO_GL=egl python scripts/render.py --robot warned \
      --checkpoint logs/.../model_*.pt --segment-frames 100 --out progression.mp4
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WANDB_MODE", "disabled")

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import brace_task  # noqa: F401,E402
from brace_task import UNWARNED_TASK, WARNED_TASK  # noqa: E402

TASKS = {"warned": WARNED_TASK, "unwarned": UNWARNED_TASK}


def _zeroed_command(env, command_name: str):
  """Stand-in for mdp.generated_commands that always reports "nothing pending".

  Same swap eval.py uses for the warning-removed condition, so the rendered clip and the measured
  number come from the identical intervention rather than two lookalikes.
  """
  return torch.zeros_like(env.command_manager.get_command(command_name))


def _font(px: int):
  """A real TrueType face if one is installed, else PIL's bitmap default.

  The default font is fixed at ~11 px, which is unreadable on a 1080p frame, so the HUD
  is worth this much effort and no more.
  """
  for path in (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
  ):
    if Path(path).exists():
      return ImageFont.truetype(path, px)
  return ImageFont.load_default()


def annotate(frame, lines, font, pad: int):
  """Draw a HUD of (text, colour) rows on a black band across the top."""
  img = Image.fromarray(frame)
  d = ImageDraw.Draw(img)
  step = font.size + pad if hasattr(font, "size") else 14 + pad
  d.rectangle([0, 0, img.width, pad + step * len(lines)], fill=(0, 0, 0))
  for i, (text, col) in enumerate(lines):
    d.text((pad, pad // 2 + step * i), text, fill=col, font=font)
  return np.asarray(img)


def state_row(ttl: float, live: bool):
  if live:
    return "IMPACT", (255, 90, 220)
  if ttl > 0:
    return f"INCOMING  t-{ttl:.2f}s", (255, 70, 60)
  return "clear", (170, 170, 170)


def iteration_of(path: str) -> int | None:
  m = re.search(r"model_(\d+)\.pt$", path)
  return int(m.group(1)) if m else None


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--robot", choices=sorted(TASKS), required=True)
  p.add_argument("--checkpoint", nargs="+", required=True,
                 help="one checkpoint, or many for a training-progression clip")
  p.add_argument("--segment-frames", type=int, default=150,
                 help="frames per checkpoint when several are given")
  p.add_argument("--no-warning", action="store_true",
                 help="feed the robot zeros instead of the warning (warned robot only)")
  p.add_argument("--force", type=float, default=160.0)
  p.add_argument("--frames", type=int, default=600,
                 help="total frames when a single checkpoint is given")
  p.add_argument("--out", default="brace.mp4")
  p.add_argument("--width", type=int, default=1920)
  p.add_argument("--height", type=int, default=1080)
  p.add_argument("--distance", type=float, default=2.0)
  p.add_argument("--elevation", type=float, default=-12.0)
  p.add_argument("--azimuth", type=float, default=120.0)
  p.add_argument("--orbit", type=float, default=0.0,
                 help="degrees of azimuth per second; 0 holds the camera still")
  p.add_argument("--crf", type=int, default=16,
                 help="x264 quality, lower is better; 16 is visually lossless")
  a = p.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer.viewer_config import ViewerConfig

  configure_torch_backends()
  dev = "cuda:0"
  task = TASKS[a.robot]

  cfg = load_env_cfg(task, play=True)
  agent = load_rl_cfg(task)
  cfg.scene.num_envs = 8
  cfg.curriculum = {}
  threat = cfg.commands["threat"]
  threat.force_range = (a.force, a.force)
  threat.force_jitter = 0.0
  threat.debug_vis = True
  # Keep the frame uncluttered: the velocity command arrows compete with the threat.
  cfg.commands["twist"].debug_vis = False

  if a.no_warning:
    if "threat" not in cfg.observations["actor"].terms:
      raise SystemExit("--no-warning only applies to the warned robot")
    cfg.observations["actor"].terms["threat"].func = _zeroed_command

  cfg.viewer.width, cfg.viewer.height = a.width, a.height
  cfg.viewer.env_idx = 0  # the only env that gets debug arrows
  cfg.viewer.max_extra_envs = 0  # one robot, cleanest read on the arrow
  cfg.viewer.enable_shadows = True
  cfg.viewer.enable_reflections = True
  # ASSET_ROOT is the origin_type that actually tracks a body; see the module docstring.
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

  cam = env._offline_renderer._cam
  assert cam.trackbodyid >= 0, "camera is not tracking; the robot will leave frame"

  robot = env.scene["robot"]
  threat_term = env.command_manager.get_term("threat")

  font = _font(max(16, a.height // 40))
  pad = max(8, a.height // 90)
  label = a.robot + (" + warning removed" if a.no_warning else "")
  many = len(a.checkpoint) > 1
  per_ckpt = a.segment_frames if many else a.frames

  # A shell glob of model_*.pt arrives in lexicographic order -- 0, 1000, 1050, ... 150
  # -- which would render the progression out of order and make it look like the policy
  # regressed. Sort by the iteration in the filename instead.
  # -1 for a name that carries no iteration, rather than None, which is not orderable
  # against an int and would crash the sort on two such names.
  ckpts = sorted(a.checkpoint, key=lambda c: iteration_of(c) if iteration_of(c) else -1)
  if many:
    print(f"progression over {len(ckpts)} checkpoints: "
          f"{iteration_of(ckpts[0])} .. {iteration_of(ckpts[-1])}")

  # Frames go straight to the encoder. The progression clip is 81 checkpoints x 100
  # frames, which at 1920x1080x3 bytes is about 50 GB if held in a list.
  enc = dict(
    fps=50, macro_block_size=1, codec="libx264",
    output_params=["-crf", str(a.crf), "-preset", "slow", "-pix_fmt", "yuv420p"],
  )
  writer = imageio.get_writer(a.out, **enc)

  n_frames = n_warn = n_impact = 0
  with torch.inference_mode():
    for ckpt in ckpts:
      runner.load(ckpt, load_cfg={"actor": True}, strict=True, map_location=dev)
      policy = runner.get_inference_policy(device=dev)
      it = iteration_of(ckpt)
      # Each checkpoint starts from a clean stance, so a fall left by the previous
      # (weaker) policy is not charged to the next one.
      obs, _ = wrapped.reset()
      for _ in range(per_ckpt):
        obs, _, _, _ = wrapped.step(policy(obs))
        f = env.render()
        if f is None:
          continue
        if a.orbit:
          cam.azimuth = a.azimuth + a.orbit * n_frames / 50.0
        ttl = float(threat_term._time_to_impact[0])
        live = bool(threat_term._sustain_left[0] > 0)
        n_warn += ttl > 0
        n_impact += live
        tilt = float(
          torch.rad2deg(
            torch.acos((-robot.data.projected_gravity_b[0, 2]).clamp(-1.0, 1.0))
          )
        )
        rows = [
          (f"Unitree Go1  |  {label}  |  {a.force:.0f} N impulse",
           (255, 255, 255)),
          state_row(ttl, live),
          (f"trunk tilt {tilt:5.1f} deg", (255, 255, 255)),
        ]
        if it is not None and many:
          rows.append((f"training iteration {it}", (120, 220, 255)))
        writer.append_data(annotate(f, rows, font, pad))
        n_frames += 1

  writer.close()
  env.close()
  print(f"wrote {n_frames} frames -> {a.out}")
  print(f"  warning visible on {n_warn} frames, impact on {n_impact}")

  # A clip that never shows a warning or never shows an impact is a failed render, not
  # a video -- fail loudly rather than leaving someone to notice it in the playback.
  assert n_warn > 0, "no warning phase was captured; raise --frames"
  assert n_impact > 0, "no impact was captured; raise --frames"


if __name__ == "__main__":
  main()
