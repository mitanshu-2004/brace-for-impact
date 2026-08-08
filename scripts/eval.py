"""Measure survival against impulses of known magnitude.

Training uses a curriculum, so force there is coupled to how good the policy already
is -- a survival-vs-force curve taken from training data measures both at once. Here
the curriculum is switched off and every environment is hit with the same commanded
force, swept across a fixed ladder. That is the number the project reports.

One row per impulse is written, not just per-force aggregates, so any later question
(tilt distribution, effect of bearing, recovery time, fall rate at a given force) is
answerable from the same file without re-running anything.

The environment is built ONCE and the force is mutated on the live command term between
levels. ThreatCommand reads cfg.force_range at sample time, so a rebuild is unnecessary
-- and a rebuild would re-pay MuJoCo Warp's JIT compile on every one of the seven
levels.

Note on pairing: the two robots do not get identical shove sequences. Attempts
end at different times when one robot falls more, which shifts every later shove.
With thousands of impulses per force level the distributions still match closely; the
comparison is statistical, not paired, and is reported that way.

Usage:
  python scripts/eval.py --robot warned --checkpoint ckpt/model_3999.pt
  python scripts/eval.py --robot warned --checkpoint ... --no-warning
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402

import brace_task  # noqa: F401,E402  (registers the tasks)
from brace_task import UNWARNED_TASK, WARNED_TASK  # noqa: E402

TASKS = {"warned": WARNED_TASK, "unwarned": UNWARNED_TASK}

# Newtons. 35 N ~ the stock push the shipped task trains against; 250 N should put an
# untrained policy down reliably. Go1 is 13.93 kg, so dv = F*dt/m.
FORCE_LADDER = (35.0, 60.0, 90.0, 120.0, 160.0, 200.0, 250.0)

# Steps to watch an impulse after it lands. 100 steps at dt=0.02 s = 2 s, long enough
# for a recoverable stumble to resolve.
WATCH_STEPS = 100


def _zeroed_command(env, command_name: str):
  """Stand-in for mdp.generated_commands that always reports "nothing pending"."""
  return torch.zeros_like(env.command_manager.get_command(command_name))


def build(task_id: str, ckpt: Path, num_envs: int, blind: bool):
  """Build the env and load the policy. Done once; force is swept afterwards."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  configure_torch_backends()
  dev = "cuda:0"

  cfg = load_env_cfg(task_id, play=True)
  agent = load_rl_cfg(task_id)
  cfg.scene.num_envs = num_envs
  cfg.commands["threat"].force_jitter = 0.0
  cfg.curriculum = {}  # the sweep sets force explicitly; no adaptive difficulty

  if blind:
    # Feed the actor zeros where the threat vector goes, leaving the impulses
    # untouched. This separates two very different null results: if survival is
    # unchanged the policy never learned to read the warning (a training failure),
    # whereas if it drops to the unwarned robot's level the warning is genuinely being
    # used and anticipation simply does not beat reacting (a real finding).
    #
    # Done by swapping the term's function rather than slicing the observation tensor:
    # no index arithmetic to get wrong, the shape is unchanged so the checkpoint still
    # loads, and all-zeros is exactly what the policy already sees between threats, so
    # this is in-distribution rather than garbage input.
    if "threat" not in cfg.observations["actor"].terms:
      raise SystemExit("--no-warning only applies to the warned robot")
    cfg.observations["actor"].terms["threat"].func = _zeroed_command

  env = ManagerBasedRlEnv(cfg=cfg, device=dev, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner = (load_runner_cls(task_id) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent), device=dev
  )
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=dev)
  return env, wrapped, runner.get_inference_policy(device=dev)


def sweep_force(env, wrapped, policy, force: float, steps: int) -> list[dict]:
  """Run one force level on an already-built env; return per-impulse rows."""
  dev = "cuda:0"
  n = env.num_envs
  robot = env.scene["robot"]
  threat = env.command_manager.get_term("threat")

  # Read at sample time by _resample_command, so mutating it here is enough for every
  # threat sampled from now on.
  threat.cfg.force_range = (force, force)

  # But threats already in flight keep the force they were sampled with, and would be
  # logged under the new label -- about one impulse per env at each boundary. Clear any
  # live wrench and resample everything so the level starts clean.
  all_ids = torch.arange(n, device=dev)
  threat._clear_wrench(all_ids)
  threat._resample(all_ids)

  live_prev = torch.zeros(n, dtype=torch.bool, device=dev)
  open_rec = torch.zeros(n, dtype=torch.bool, device=dev)
  rec_bearing = torch.zeros(n, device=dev)
  rec_peak_tilt = torch.zeros(n, device=dev)
  rec_age = torch.zeros(n, dtype=torch.long, device=dev)
  rec_fell = torch.zeros(n, dtype=torch.bool, device=dev)
  rec_start_xy = torch.zeros(n, 2, device=dev)
  rec_disp = torch.zeros(n, device=dev)

  rows: list[dict] = []

  def flush(mask: torch.Tensor) -> None:
    ids = mask.nonzero(as_tuple=False).flatten()
    if len(ids) == 0:
      return
    for i in ids.tolist():
      rows.append(
        dict(
          force_n=force,
          bearing_rad=round(float(rec_bearing[i]), 4),
          peak_tilt_deg=round(float(torch.rad2deg(rec_peak_tilt[i])), 3),
          displacement_m=round(float(rec_disp[i]), 4),
          watched_steps=int(rec_age[i]),
          fell=bool(rec_fell[i]),
        )
      )
    open_rec[ids] = False

  obs = wrapped.get_observations()
  with torch.inference_mode():
    for _ in range(steps):
      obs, _, dones, _ = wrapped.step(policy(obs))

      tilt = torch.acos((-robot.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
      xy = robot.data.root_link_pos_w[:, :2]
      live = threat._sustain_left > 0.0

      # An episode ending while a record is open is a fall attributable to the impulse
      # being watched. `dones` includes timeouts, which are not falls, so the failure
      # flag is tested specifically.
      failed = env.termination_manager.terminated
      rec_fell |= open_rec & failed
      flush(open_rec & dones.bool())

      still = open_rec
      rec_age[still] += 1
      torch.maximum(rec_peak_tilt, tilt, out=rec_peak_tilt)
      rec_disp[still] = torch.norm(xy[still] - rec_start_xy[still], dim=-1)
      flush(still & (rec_age >= WATCH_STEPS))

      # Rising edge of the applied wrench = a fresh impact.
      landed = live & ~live_prev & ~open_rec
      if landed.any():
        ids = landed.nonzero(as_tuple=False).flatten()
        open_rec[ids] = True
        rec_bearing[ids] = threat._theta_w[ids]
        rec_peak_tilt[ids] = tilt[ids]
        rec_age[ids] = 0
        rec_fell[ids] = False
        rec_start_xy[ids] = xy[ids]
        rec_disp[ids] = 0.0
      live_prev = live.clone()

  return rows


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--robot", choices=sorted(TASKS), required=True)
  p.add_argument("--checkpoint", required=True)
  p.add_argument("--num-envs", type=int, default=1024)
  p.add_argument("--steps", type=int, default=1500)
  p.add_argument("--out", default=None)
  p.add_argument(
    "--no-warning",
    action="store_true",
    help="warned robot only: zero the warning to test whether it is really being used "
    "actually uses the warning",
  )
  a = p.parse_args()

  ckpt = Path(a.checkpoint)
  if not ckpt.exists():
    raise SystemExit(f"checkpoint not found: {ckpt}")

  label = "warning_removed" if a.no_warning else a.robot
  print(f"building env once for {label} ({a.num_envs} envs)...", flush=True)
  env, wrapped, policy = build(TASKS[a.robot], ckpt, a.num_envs, a.no_warning)

  all_rows: list[dict] = []
  for force in FORCE_LADDER:
    rows = sweep_force(env, wrapped, policy, force, a.steps)
    for r in rows:
      r["robot"] = label
    all_rows.extend(rows)
    fell = sum(r["fell"] for r in rows)
    tilt = sum(r["peak_tilt_deg"] for r in rows) / max(len(rows), 1)
    print(
      f"  {force:6.0f} N  impulses={len(rows):6d}  fell={fell:5d} "
      f"({100 * fell / max(len(rows), 1):5.1f}%)  mean peak tilt={tilt:5.1f} deg",
      flush=True,
    )
  env.close()

  if not all_rows:
    raise SystemExit("no impulses recorded -- raise --steps")

  out = Path(a.out or f"eval_{label}.csv")
  with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
    w.writeheader()
    w.writerows(all_rows)
  print(f"\nwrote {len(all_rows)} impulse rows -> {out}")


if __name__ == "__main__":
  main()
