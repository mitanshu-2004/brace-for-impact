"""Tests for the threat command and the two robot configurations.

The configuration tests run anywhere. The wrench lifecycle test needs CUDA and is
skipped without it.

  pytest                    # everything available on this machine
  pytest -m "not gpu"       # configuration only
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("WANDB_MODE", "disabled")

import brace_task  # noqa: E402,F401  (import registers the tasks)
from brace_task import UNWARNED_TASK, WARNED_TASK, brace_env_cfg  # noqa: E402
from brace_task.threat_command import ThreatCommandCfg  # noqa: E402

requires_gpu = pytest.mark.skipif(
  not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.fixture(scope="module")
def configs():
  return brace_env_cfg(warned=True), brace_env_cfg(warned=False)


def test_both_tasks_register():
  from mjlab.tasks.registry import list_tasks

  tasks = list_tasks()
  assert WARNED_TASK in tasks
  assert UNWARNED_TASK in tasks


def test_robots_differ_only_by_the_threat_observation(configs):
  """The comparison is only valid if this is the single difference between them."""
  warned, unwarned = configs
  a_actor = set(warned.observations["actor"].terms)
  r_actor = set(unwarned.observations["actor"].terms)
  assert a_actor - r_actor == {"threat"}
  assert r_actor - a_actor == set()


def test_critic_sees_the_threat_for_both_robots(configs):
  """Otherwise the unwarned robot is handicapped twice: worse input and worse advantages."""
  for cfg in configs:
    assert "threat" in cfg.observations["critic"].terms


def test_stock_push_is_removed(configs):
  """mjlab's push_robot is an untelegraphed, mass-independent velocity kick.

  Leaving it in would add a second disturbance that neither robot can see coming.
  """
  for cfg in configs:
    assert "push_robot" not in cfg.events


def test_anti_collapse_termination_and_curriculum_present(configs):
  warned, _ = configs
  assert "collapsed" in warned.terminations
  assert "threat_levels" in warned.curriculum


def test_config_rejects_a_resample_window_that_can_overlap():
  """A new threat must not be sampled while the previous wrench is still live."""
  with pytest.raises(ValueError):
    ThreatCommandCfg(
      resampling_time_range=(1.0, 2.0),  # below warn_max + duration_max = 1.5
      warn_range=(0.4, 1.2),
      duration_range=(0.1, 0.3),
    )


@pytest.mark.gpu
@requires_gpu
def test_wrench_lifecycle():
  """Advertise, apply, clear, and do not survive a reset.

  The reset case is the one worth a test: xfrc_applied persists until something writes
  zeros, so a wrench live at episode end would carry into the next episode and apply a
  force nothing asked for.
  """
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.utils.torch import configure_torch_backends

  num_envs, steps = 64, 600  # 600 steps at dt=0.02 is 12 s, several threat cycles

  configure_torch_backends()
  cfg = load_env_cfg(WARNED_TASK)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0", render_mode=None)

  robot = env.scene["robot"]
  threat = env.command_manager.get_term("threat")
  trunk_id = threat._body_ids[0]

  saw_pending = saw_impact = saw_clear = False
  leaked_after_reset = 0
  prev_dones = torch.zeros(num_envs, dtype=torch.bool, device="cuda:0")
  zero_action = torch.zeros((num_envs, env.action_space.shape[-1]), device="cuda:0")

  try:
    with torch.inference_mode():
      for _ in range(steps):
        env.step(zero_action)

        fmag = robot.data.body_external_wrench[:, trunk_id, :3].norm(dim=-1)
        cmd = env.command_manager.get_command("threat")

        if prev_dones.any():
          leaked_after_reset += int((fmag[prev_dones] > 1e-6).sum())

        if (cmd[:, 3] > 0).any():
          saw_pending = True
        if (fmag > 1e-6).any():
          saw_impact = True
          live = fmag > 1e-6
          assert float(cmd[live, 3].abs().max()) < 1e-6, (
            "command advertises a pending threat while the wrench is already applied"
          )
        if saw_impact and (fmag < 1e-6).all():
          saw_clear = True

        prev_dones = env.termination_manager.dones.clone()
  finally:
    env.close()

  assert saw_pending, "no threat was advertised in the command vector"
  assert saw_impact, "no impulse was applied"
  assert saw_clear, "the wrench was never cleared after an impulse expired"
  assert leaked_after_reset == 0, (
    f"{leaked_after_reset} env-steps had a live wrench immediately after a reset"
  )
