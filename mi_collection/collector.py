"""Env stack construction and the batched rollout loop.

The collector builds its own env stack instead of reusing
``env_runner/vistac_pih_runner_unit.py``.  That duplicates ~20 lines, but those
runners are what produced the reported eval numbers and are left untouched;
the collector also needs a different stack (a recorder inside ``MultiStepWrapper``,
no video wrapper, no ``max_episode_steps`` truncation).

Episodes run in parallel -- ``num_envs`` of them per batch, one per env slot --
and seeding is global, so an episode's identity is ``(batch_seed, env_index)``.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply

from manifeel.envs.vistac_isaacgym_multiple_env_wrapper import MultipleIsaacEnvWrapper
from manifeel.gym_util.multistep_wrapper import MultiStepWrapper

from mi_collection.contact import build_pair_indices, sidecar_contact_pairs
from mi_collection.episode_accumulator import BatchAccumulator
from mi_collection.recording_wrapper import (
    DEFAULT_OBS_CHANNELS,
    InnerStepRecorder,
    validate_render_cache,
)
from mi_collection import writer as writer_mod

#: Keys the runner resizes before handing the obs to the policy.  Recording is
#: unaffected -- it reads ``render_cache``, which is set before any of this.
_TACTILE_RGB_KEYS = ("left_tactile_camera_taxim", "right_tactile_camera_taxim")


class CollectOptions:
    """Everything the driver can vary between runs."""

    def __init__(
        self,
        out_dir: str,
        num_episodes: int = 1000,
        num_envs: int = 50,
        start_seed: int = 300000,
        max_steps: int = 500,
        success_tail_steps: int = 20,
        contact_threshold: float = 1e-3,
        include_optional_pairs: bool = True,
        obs_channels: Optional[Dict[str, str]] = None,
        resume: bool = False,
        keep_shards: bool = False,
    ) -> None:
        self.out_dir = os.path.expanduser(out_dir)
        self.num_episodes = int(num_episodes)
        self.num_envs = int(num_envs)
        self.start_seed = int(start_seed)
        self.max_steps = int(max_steps)
        self.success_tail_steps = int(success_tail_steps)
        self.contact_threshold = float(contact_threshold)
        self.include_optional_pairs = bool(include_optional_pairs)
        self.obs_channels = dict(
            DEFAULT_OBS_CHANNELS if obs_channels is None else obs_channels
        )
        self.resume = bool(resume)
        self.keep_shards = bool(keep_shards)

    @property
    def num_batches(self) -> int:
        return int(np.ceil(self.num_episodes / self.num_envs))

    def shard_params(self) -> Dict[str, object]:
        """The parameters a reusable shard must match, for ``--resume``."""
        return {
            "num_envs": self.num_envs,
            "max_steps": self.max_steps,
            "success_tail_steps": self.success_tail_steps,
            "contact_threshold": self.contact_threshold,
            "include_optional_pairs": self.include_optional_pairs,
            "obs_channels": dict(sorted(self.obs_channels.items())),
        }


class MICollector:
    """Rolls a trained policy out and writes one MI shard per batch."""

    def __init__(
        self,
        cfg: Any,
        policy: Any,
        device: torch.device,
        task_spec: Any,
        options: CollectOptions,
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.device = device
        self.task_spec = task_spec
        self.options = options
        self.env = None
        self.inner = None
        self.recorder = None
        self.pair_indices: List[Any] = []
        runner_cfg = cfg.task.env_runner
        self.n_obs_steps = int(cfg.n_obs_steps)
        self.n_action_steps = int(cfg.n_action_steps)
        self.tactile_size = list(runner_cfg.get("tactile_size", [256, 256]))
        self.shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
        self.isaacgym_cfg_name = str(runner_cfg.isaacgym_cfg_name)

    # -- construction ------------------------------------------------------
    def build(self) -> None:
        isaacgym_cfg = hydra.compose(config_name=self.isaacgym_cfg_name)
        isaacgym_cfg["shape_meta"] = OmegaConf.create(self.shape_meta)
        _select_gpu(isaacgym_cfg)
        isaacgym_cfg.num_envs = self.options.num_envs

        self.inner = MultipleIsaacEnvWrapper(isaacgym_cfg)
        task = self.inner.envs
        if getattr(task, "contact_force_pairwise", None) is None:
            raise RuntimeError(
                "contact_force_pairwise is None: this sim is on the CPU pipeline, "
                "where Isaac Gym exposes neither the pairwise contact tensor nor "
                "the tactile force field. Nothing would be recorded but zeros. "
                "Run with pipeline=gpu."
            )
        self.pair_indices = build_pair_indices(
            task, self.task_spec, self.options.include_optional_pairs
        )
        self.recorder = InnerStepRecorder(
            env=self.inner,
            inner_env=self.inner,
            task_spec=self.task_spec,
            pair_indices=self.pair_indices,
            control_dt=self.control_dt,
            contact_threshold=self.options.contact_threshold,
            obs_channels=self.options.obs_channels,
        )
        # No truncation here: episode length is decided per env by the
        # accumulator, and `reset_buf` is a success flag, not a boundary.
        self.env = MultiStepWrapper(
            self.recorder,
            n_obs_steps=self.n_obs_steps,
            n_action_steps=self.n_action_steps,
            max_episode_steps=None,
        )

    @property
    def control_dt(self) -> float:
        """One control step in sim seconds: ``sim.dt * controlFrequencyInv``."""
        task = self.inner.envs
        return float(task.cfg_base.sim.dt) * float(
            task.cfg_task.env.controlFrequencyInv
        )

    # -- rollout -----------------------------------------------------------
    def run(self) -> Dict[str, object]:
        if self.env is None:
            self.build()
        options = self.options
        os.makedirs(os.path.join(options.out_dir, "shards"), exist_ok=True)
        params = options.shard_params()
        # Only when the spec has any, so peg-insertion shard params are unchanged.
        spec_params = self.task_spec.collection_params()
        if spec_params:
            params["task_spec"] = spec_params
        shard_paths: List[str] = []
        batch_reports: List[Dict[str, object]] = []

        for batch in range(options.num_batches):
            path = writer_mod.shard_path(options.out_dir, batch)
            seed = options.start_seed + batch * options.num_envs
            if options.resume and writer_mod.shard_is_reusable(path, params):
                info = writer_mod.read_json(
                    os.path.join(path, writer_mod.SHARD_INFO_BASENAME)
                )
                print(f"[batch {batch}] reusing complete shard {path}")
                shard_paths.append(path)
                batch_reports.append(info)
                continue
            writer_mod.discard_shard(path)
            report = self._run_batch(batch, seed, path, params)
            shard_paths.append(path)
            batch_reports.append(report)

        return {
            "shard_paths": shard_paths,
            "batches": batch_reports,
            "contact_pairs": sidecar_contact_pairs(self.pair_indices),
            "control_dt": self.control_dt,
        }

    def _run_batch(
        self, batch: int, seed: int, path: str, params: Dict[str, object]
    ) -> Dict[str, object]:
        options = self.options
        shard = writer_mod.ShardWriter(path, batch, seed)
        accumulator = BatchAccumulator(
            num_envs=options.num_envs,
            max_steps=options.max_steps,
            tail_steps=options.success_tail_steps,
            on_complete=lambda _i, episode, meta: shard.add(episode, meta),
            annotate=lambda episode: self.task_spec.annotate_episode(
                episode, self.control_dt
            ),
        )
        self.recorder.sink = accumulator.push

        self.inner.seed(seed)
        obs = self.env.reset()
        self.policy.reset()
        missing = validate_render_cache(self.inner, list(options.obs_channels))
        if missing:
            raise KeyError(
                f"Channels {missing} are not produced by this task; "
                f"have {sorted(self.inner.render_cache)}."
            )

        started = time.time()
        ceiling = accumulator.physical_ceiling
        # A 500-step batch is minutes of silence otherwise, which is
        # indistinguishable from a hang.
        report_every = max(self.n_action_steps, ceiling // 10)
        next_report = report_every
        while not accumulator.done and self.recorder.step_index < ceiling:
            action = self._predict(obs)
            obs, _reward, _done, _info = self.env.step(action)
            if self.recorder.step_index >= next_report:
                print(
                    f"  [batch {batch}] step {self.recorder.step_index}/{ceiling}  "
                    f"finished {accumulator.completed}/{options.num_envs} envs  "
                    f"{time.time() - started:.0f}s"
                )
                next_report += report_every
        accumulator.finish()

        elapsed = time.time() - started
        info = shard.finalize(params)
        successes = int(
            np.sum([row["success"] for row in shard.meta_rows], dtype=np.int64)
        )
        lengths = [row["length"] for row in shard.meta_rows]
        info.update(
            {
                "wall_clock_s": round(elapsed, 1),
                "steps_driven": int(self.recorder.step_index),
                "num_success": successes,
                "mean_episode_length": float(np.mean(lengths)),
            }
        )
        print(
            f"[batch {batch}] seed={seed} envs={options.num_envs} "
            f"success={successes}/{options.num_envs} "
            f"steps_driven={self.recorder.step_index} "
            f"mean_len={np.mean(lengths):.1f} {elapsed:.0f}s -> {path}"
        )
        return info

    def _predict(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        np_obs = dict(obs)
        for key in _TACTILE_RGB_KEYS:
            if key in np_obs:
                np_obs[key] = _resize_tactile(np_obs[key], self.tactile_size)
        obs_dict = dict_apply(np_obs, lambda x: torch.from_numpy(x).to(self.device))
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
        return action_dict["action"].detach().to("cpu").numpy()


def _resize_tactile(data: np.ndarray, size: List[int]) -> np.ndarray:
    """``(B, T, C, H, W)`` -> ``(B, T, C, size[0], size[1])``, as the runner does."""
    b_dim, t_dim, c_dim = data.shape[:3]
    out = np.zeros((b_dim, t_dim, c_dim, size[0], size[1]), dtype=data.dtype)
    for b in range(b_dim):
        for t in range(t_dim):
            for c in range(c_dim):
                out[b, t, c] = cv2.resize(
                    data[b, t, c], (size[1], size[0]), interpolation=cv2.INTER_LINEAR
                )
    return out


def _select_gpu(isaacgym_cfg: Any) -> None:
    """Point the sim at ``$MANIFEEL_GPU`` -- see the runner for why not CUDA_VISIBLE_DEVICES."""
    gpu = os.environ.get("MANIFEEL_GPU")
    if not gpu:
        return
    isaacgym_cfg.sim_device = f"cuda:{gpu}"
    isaacgym_cfg.rl_device = f"cuda:{gpu}"
    isaacgym_cfg.graphics_device_id = int(gpu)
    torch.cuda.set_device(int(gpu))
