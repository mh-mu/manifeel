"""Inner-step recording hook.

Sits **inside** ``MultiStepWrapper``: that wrapper's ``step()`` loops over
``n_action_steps`` inner env steps, so a hook outside it would store one row per
policy call instead of one per control step.

Alignment follows the ShapeFilter collector
(``mutual_information/dataloading/collect_mi_dataset.py``): the observation and
the contact descriptor are read **before** the action, the action is the one
taken **at** that step, and reward/success are the ones **returned by** it.

The wrapper also masks ``done`` on the way up.  In this task ``reset_buf`` is a
per-step success flag, not an episode boundary -- there is no auto-reset -- and
``MultiStepWrapper`` would break out of its inner loop (and then out of every
later call, permanently) the moment every env happened to be successful at once.
Episode completion is tracked by :mod:`mi_collection.episode_accumulator`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import gym
import numpy as np

from mi_collection.contact import contact_descriptors

#: How each recorded observation channel is converted out of ``render_cache``.
IMAGE_U8 = "image_u8"  # (E, 3, H, W) in [0,1] or uint8 -> uint8 CHW
SPATIAL_F32 = "spatial_f32"  # (E, C, H, W) -> float32 (H, W, C)
LOWDIM_F32 = "lowdim_f32"  # (E, D) -> float32 (D,)

#: Default recorded observation channels: the vision side, the two tactile
#: sides, and the policy's own low-dim state.  ``side``/``wrist_2``/the left
#: tactile sensors are skipped -- tactile RGB compresses ~7x worse than a
#: rendered view, so they dominate storage for modalities no policy consumes.
DEFAULT_OBS_CHANNELS: Dict[str, str] = {
    "wrist": IMAGE_U8,
    "front": IMAGE_U8,
    "right_tactile_camera_taxim": IMAGE_U8,
    "tactile_force_field_right": SPATIAL_F32,
    "state": LOWDIM_F32,
}


def to_uint8(array: np.ndarray) -> np.ndarray:
    """Image to uint8, accepting either a ``[0,1]`` float or an existing uint8."""
    if array.dtype == np.uint8:
        return array
    scaled = np.rint(np.asarray(array, dtype=np.float32) * 255.0)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def to_spatial_f32(array: np.ndarray) -> np.ndarray:
    """``(E, C, H, W)`` back to the ``(E, H, W, C)`` layout the sim produced.

    ``MultipleIsaacEnvWrapper._transform_obs_data`` moves the trailing size-3
    axis to the front for every obs whose last dimension is 3, which catches the
    tactile force field as well as the cameras.  Storing it shaped (rather than
    flattened) keeps the taxel grid, and the analyzer can flatten later.
    """
    return np.ascontiguousarray(np.moveaxis(array, 1, -1)).astype(np.float32)


class InnerStepRecorder(gym.Wrapper):
    """Records one row per inner control step and hands it to ``sink``."""

    def __init__(
        self,
        env: object,
        inner_env: object,
        task_spec: object,
        pair_indices: Sequence[object],
        control_dt: float,
        contact_threshold: float,
        obs_channels: Optional[Dict[str, str]] = None,
        sink: Optional[Callable[[int, Dict[str, np.ndarray]], None]] = None,
        mask_done: bool = True,
    ) -> None:
        super().__init__(env)
        # Held explicitly rather than discovered by walking `.env`: the stack
        # below this wrapper is the collector's business, not this class's.
        self.inner_env = inner_env
        self.task_spec = task_spec
        self.pair_indices = list(pair_indices)
        self.control_dt = float(control_dt)
        self.contact_threshold = float(contact_threshold)
        self.obs_channels = dict(
            DEFAULT_OBS_CHANNELS if obs_channels is None else obs_channels
        )
        self.sink = sink
        self.mask_done = mask_done
        self.step_index = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.step_index = 0
        return obs

    def step(self, action):
        row = self._snapshot_pre_step()
        row["action"] = np.asarray(action, dtype=np.float32)
        row["timestamp"] = np.full(
            (self._num_envs(), 1), self.step_index * self.control_dt, dtype=np.float32
        )

        obs, reward, done, info = self.env.step(action)

        task = self.inner_env.envs
        row["reward_total"] = (
            task.rew_buf.detach().to("cpu").numpy().astype(np.float32).reshape(-1, 1)
        )
        row["success"] = self.task_spec.success(task).reshape(-1, 1).astype(np.uint8)

        if self.sink is not None:
            self.sink(self.step_index, row)
        self.step_index += 1

        if self.mask_done:
            done = np.zeros_like(np.asarray(done))
        return obs, reward, done, info

    # -- internals ---------------------------------------------------------
    def _num_envs(self) -> int:
        return int(self.inner_env.num_envs)

    def _snapshot_pre_step(self) -> Dict[str, np.ndarray]:
        """Observations, contacts and world state of the pre-action sim state."""
        cache = self.inner_env.render_cache
        if not isinstance(cache, dict) or not cache:
            raise RuntimeError(
                "render_cache is empty -- reset() must run before the first step."
            )
        row: Dict[str, np.ndarray] = {}
        for channel, kind in self.obs_channels.items():
            if channel not in cache:
                raise KeyError(
                    f"Channel {channel!r} is not in render_cache; "
                    f"have {sorted(cache)}."
                )
            row[channel] = _convert(cache[channel], kind)

        task = self.inner_env.envs
        descriptors = contact_descriptors(
            task.contact_force_pairwise, self.pair_indices, self.contact_threshold
        )
        host = descriptors.detach().to("cpu").numpy().astype(np.float32)
        for index, pair in enumerate(self.pair_indices):
            row[pair.name] = host[:, index]
        row["state_full"] = self.task_spec.state_full(task)
        return row


def _convert(value: np.ndarray, kind: str) -> np.ndarray:
    if kind == IMAGE_U8:
        return to_uint8(value)
    if kind == SPATIAL_F32:
        return to_spatial_f32(value)
    if kind == LOWDIM_F32:
        return np.ascontiguousarray(value).astype(np.float32)
    raise ValueError(f"Unknown channel kind {kind!r}.")


def validate_render_cache(inner_env: object, channels: Sequence[str]) -> List[str]:
    """Names of ``channels`` the task does not produce.  Empty means all present.

    ``_apply_obs_by_keys`` filters the observation without warning, so a channel
    that silently disappears has to be caught before a batch runs rather than
    found as a zero column afterwards.
    """
    cache = inner_env.render_cache
    return [name for name in channels if name not in cache]
