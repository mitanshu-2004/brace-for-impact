# Notes on mjlab 1.5.3

Behaviour of the released wheel that is either undocumented or differs from what the
source on GitHub suggests. Collected while building this task; useful to anyone else
writing a `CommandTerm` or rendering offscreen.

Everything below was checked against `mjlab==1.5.3` installed from PyPI.

## The released wheel and GitHub main are not the same code

They share a version string. They do not share the `CommandTerm` contract.

On main, `CommandTerm.compute` is `compute(self, dt, env_ids=None)` and calls
`self._update_command(env_ids)`, with a `_check_update_command_signature` that requires
the parameter. In the released 1.5.3 the signature is `compute(self, dt)` and the call is
`self._update_command()` with no arguments.

A term written against main raises `TypeError: _update_command() missing 1 required
positional argument` under the release. Diffing the two files, that is the only
difference in `command_manager.py`.

A term that works under both defaults the parameter and advances time only when it is
absent:

```python
def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
  if env_ids is None:          # per-step path on both lines
    self._advance(self._env.step_dt)
  self._refresh_command()      # reset path on main passes ids with dt = 0.0
```

Pin the version. A range makes which contract applies depend on when you installed.

## External wrenches persist until overwritten

`write_external_wrench_to_sim(forces, torques, env_ids, body_ids)` writes into
`xfrc_applied`, in the world frame. It stays there across steps until something writes
zeros.

Two consequences. A timed impulse has to be explicitly cleared when its duration expires,
and an active wrench has to be cleared on reset, or it carries into the next episode.
mjlab's own `apply_body_impulse` event clears on reset for the same reason.

## `ViewerConfig.origin_type` AUTO does not track

The enum member's docstring reads "Track the first non-fixed body, or fall back to a free
camera." `_setup_camera()` handles AUTO in the same branch as WORLD:

```python
if self._cfg.origin_type in (OriginType.AUTO, OriginType.WORLD):
  camera.type = mujoco.mjtCamera.mjCAMERA_FREE.value
  camera.fixedcamid = -1
  camera.trackbodyid = -1
```

The result is a fixed camera at the world origin. Anything that moves leaves the frame.
`ASSET_ROOT` with `entity_name`, or `ASSET_BODY` with `entity_name` and `body_name`, are
the branches that set `trackbodyid`.

Worth asserting `cam.trackbodyid >= 0` after building the environment if you expect
tracking, since nothing else reports it.

## Camera pose does come from the cfg

`_setup_camera()` assigns `lookat`, `elevation`, `azimuth`, and `distance` from the config
at the end of the function. Setting them on `renderer._cam` afterwards works but skips
`_compute_render_extent()`, which derives the render extent from `cfg.distance`. MuJoCo
scales z-near, z-far, and the shadow clip off `model.stat.extent`, so the pose and the
extent end up disagreeing. Set the pose on the config.

(This differs from main, where the pose fields are ignored by the offscreen renderer.)

## Debug visuals reach the offscreen renderer, for one environment

`ManagerBasedRlEnv` passes a `debug_vis_callback` into the offscreen renderer, so arrows
drawn in `_debug_vis_impl` appear in rendered video and not only in the interactive
viewer.

`update()` constructs `MujocoNativeDebugVisualizer` without `show_all_envs`, so only
`cfg.viewer.env_idx` is annotated. Other environments render normally but without
overlays. Set `max_extra_envs = 0` for a single annotated robot.

## The offscreen framebuffer is sized from the config

`OffscreenRenderer.__init__` writes `cfg.viewer.width` and `cfg.viewer.height` into
`model.vis.global_.offwidth/offheight`. A standalone `mujoco.Renderer` on a
hand-written model will refuse a size above the model's default 640x480 with
`Image width N > framebuffer width 640`; that error comes from the model, not from EGL.

## Rendering needs an NVIDIA EGL vendor file

`MUJOCO_GL=egl` requires `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`. Images that ship
only `50_mesa.json`, including Kaggle's, cannot create a context on the GPU.

An `EGLError` raised from a `__del__` during interpreter shutdown is noise. The context is
being freed after the display is gone, and it appears after the render has finished and
written its output.

## imageio-ffmpeg has no drawtext filter

The bundled ffmpeg binary is built without libfreetype, so `drawtext` does not exist and a
filter graph using it fails to parse with `No such filter: 'drawtext'`. `overlay` is
available, so captions can be rendered to a PNG and composited instead.
