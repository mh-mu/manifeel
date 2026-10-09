"""Shard zarr writing, consolidation and the ``mi_collect_info.json`` sidecar.

One shard per batch, consolidated at the end.  Appending straight to the final
store would leave it with an inconsistent ``data``/``episode_ends`` length if a
batch died mid-episode; a shard is either complete (it has a
``shard_info.json`` with ``"complete": true``) or it is thrown away and redone.

Layout is the diffusion_policy ``ReplayBuffer`` one the fbsource MI analyzers
already read: ``data/<channel>`` concatenated along axis 0 plus
``meta/episode_ends``.  The extra ``meta`` arrays (``episode_success`` and
friends) are what let MI be computed over success-only or failure-only rollouts.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Dict, List, Optional, Sequence

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

SHARD_INFO_BASENAME: str = "shard_info.json"
COLLECT_INFO_BASENAME: str = "mi_collect_info.json"

#: Per-episode ``meta`` arrays, in the order they are written.
EPISODE_META_KEYS: Sequence[str] = (
    "episode_success",
    "first_success_step",
    "episode_batch_seed",
    "episode_env_index",
)


class ShardWriter:
    """Appends episodes to one batch's zarr and records its per-episode meta."""

    def __init__(self, path: str, batch_index: int, batch_seed: int) -> None:
        self.path = os.path.expanduser(path)
        self.batch_index = int(batch_index)
        self.batch_seed = int(batch_seed)
        self.buffer = ReplayBuffer.create_from_path(self.path, mode="a")
        self.meta_rows: List[Dict[str, int]] = []

    def add(self, episode: Dict[str, np.ndarray], meta: Dict[str, int]) -> None:
        self.buffer.add_episode(episode, compressors="disk")
        self.meta_rows.append(dict(meta))

    def finalize(self, params: Dict[str, object]) -> Dict[str, object]:
        """Write the per-episode meta arrays and the shard's completion marker."""
        order = np.argsort(
            np.asarray([row["env_index"] for row in self.meta_rows], dtype=np.int64),
            kind="stable",
        )
        self.buffer.update_meta(
            {
                "episode_success": np.asarray(
                    [self.meta_rows[i]["success"] for i in order], dtype=np.uint8
                ),
                "first_success_step": np.asarray(
                    [self.meta_rows[i]["first_success_step"] for i in order],
                    dtype=np.int64,
                ),
                "episode_env_index": np.asarray(
                    [self.meta_rows[i]["env_index"] for i in order], dtype=np.int64
                ),
                # Seeding is global (set_seed touches numpy/torch), so an episode
                # is identified by (batch seed, env slot), never by its own seed.
                "episode_batch_seed": np.full(
                    len(order), self.batch_seed, dtype=np.int64
                ),
                # Completion order, so a shard can be replayed exactly.
                "episode_write_order": np.asarray(order, dtype=np.int64),
            }
        )
        info = {
            "complete": True,
            "batch_index": self.batch_index,
            "batch_seed": self.batch_seed,
            "num_episodes": int(self.buffer.n_episodes),
            "num_steps": int(self.buffer.n_steps),
            "channels": sorted(self.buffer.keys()),
            "params": dict(params),
        }
        write_json(os.path.join(self.path, SHARD_INFO_BASENAME), info)
        return info


def write_json(path: str, payload: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str) -> Optional[Dict[str, object]]:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def shard_path(out_dir: str, batch_index: int) -> str:
    return os.path.join(out_dir, "shards", f"batch_{batch_index:04d}.zarr")


def shard_is_reusable(path: str, params: Dict[str, object]) -> bool:
    """True when ``path`` holds a complete shard collected with ``params``.

    An incomplete or parameter-mismatched shard is not reused: silently
    stitching two horizons (or two contact thresholds) into one dataset would
    produce a number no one could trace.
    """
    info = read_json(os.path.join(path, SHARD_INFO_BASENAME))
    if info is None or not info.get("complete"):
        return False
    return info.get("params") == params


def discard_shard(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def consolidate(
    shard_paths: Sequence[str],
    out_path: str,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """Concatenate complete shards into one store, in ``(batch, env_index)`` order.

    ``limit`` trims the tail: ``num_episodes`` need not divide ``num_envs``, and
    the last batch still runs a full set of env slots.
    """
    out_path = os.path.expanduser(out_path)
    if os.path.exists(out_path):
        if not overwrite:
            raise FileExistsError(
                f"{out_path} exists; pass overwrite=True to replace it."
            )
        shutil.rmtree(out_path)
    final = ReplayBuffer.create_from_path(out_path, mode="a")
    meta: Dict[str, List[int]] = {key: [] for key in EPISODE_META_KEYS}
    written = 0
    for path in shard_paths:
        if limit is not None and written >= limit:
            break
        shard = ReplayBuffer.create_from_path(path, mode="r")
        order = np.asarray(shard.meta["episode_write_order"][:], dtype=np.int64)
        take = len(order) if limit is None else min(len(order), limit - written)
        # `order` is the completion-order index of each slot in env order, and
        # the shard's meta arrays are already stored in env order -- so the meta
        # is read straight through while the data has to be de-permuted.
        for key in EPISODE_META_KEYS:
            meta[key].extend(np.asarray(shard.meta[key][:take]).tolist())
        for slot in order[:take]:
            final.add_episode(
                shard.get_episode(int(slot), copy=True), compressors="disk"
            )
        written += take
    final.update_meta(
        {
            "episode_success": np.asarray(meta["episode_success"], dtype=np.uint8),
            "first_success_step": np.asarray(
                meta["first_success_step"], dtype=np.int64
            ),
            "episode_batch_seed": np.asarray(
                meta["episode_batch_seed"], dtype=np.int64
            ),
            "episode_env_index": np.asarray(meta["episode_env_index"], dtype=np.int64),
        }
    )
    return {
        "num_episodes": int(final.n_episodes),
        "num_steps": int(final.n_steps),
        "channels": sorted(final.keys()),
        "num_success": int(np.sum(meta["episode_success"])),
    }


def write_sidecar(out_path: str, info: Dict[str, object]) -> str:
    """Write ``mi_collect_info.json`` beside the consolidated store."""
    path = os.path.join(os.path.expanduser(out_path), COLLECT_INFO_BASENAME)
    write_json(path, info)
    return path


def channel_roles(channels: Sequence[str]) -> Dict[str, str]:
    """Tag each channel sensor / world_state / contact / reward / action / time.

    Same vocabulary as the ShapeFilter collector's ``_channel_roles`` so the
    fbsource readers see a familiar sidecar.
    """
    roles: Dict[str, str] = {}
    for name in channels:
        if name == "state_full":
            roles[name] = "world_state"
        elif name == "action":
            roles[name] = "action"
        elif name == "timestamp":
            roles[name] = "time"
        elif name in ("success", "success_upstream"):
            roles[name] = "outcome"
        elif name in ("motion_primitive", "primitive_phase"):
            roles[name] = "annotation"
        elif name.startswith("contact"):
            roles[name] = "contact"
        elif name.startswith("reward"):
            roles[name] = "reward"
        else:
            roles[name] = "sensor"
    return roles
