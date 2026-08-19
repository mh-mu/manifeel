"""Pilot statistics: the numbers that pick ``max_steps`` and size the full run.

Reads a collected store (consolidated or a single shard) and reports success
rate, first-success quantiles, compressed bytes per channel and per episode, and
the projection to a target episode count.  Pick **one** common reduced
``max_steps`` across all three checkpoints from these -- a per-checkpoint horizon
would confound the sensing comparison.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

QUANTILES: Sequence[int] = (0, 50, 90, 95, 99, 100)


def directory_bytes(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def channel_bytes(store_path: str) -> Dict[str, int]:
    data_dir = os.path.join(store_path, "data")
    if not os.path.isdir(data_dir):
        return {}
    return {
        name: directory_bytes(os.path.join(data_dir, name))
        for name in sorted(os.listdir(data_dir))
        if os.path.isdir(os.path.join(data_dir, name))
    }


def summarize(store_path: str, project_to: Optional[int] = None) -> Dict[str, object]:
    buffer = ReplayBuffer.create_from_path(store_path, mode="r")
    ends = np.asarray(buffer.episode_ends[:], dtype=np.int64)
    lengths = np.diff(np.concatenate([[0], ends]))
    success = _meta(buffer, "episode_success", lengths.size, np.uint8)
    first = _meta(buffer, "first_success_step", lengths.size, np.int64, fill=-1)
    solved = first[first >= 0]

    per_channel = channel_bytes(store_path)
    total_bytes = directory_bytes(store_path)
    n_episodes = int(lengths.size)
    report: Dict[str, object] = {
        "store": store_path,
        "num_episodes": n_episodes,
        "num_steps": int(ends[-1]) if ends.size else 0,
        "success_rate": float(np.mean(success)) if success.size else 0.0,
        "num_success": int(np.sum(success)),
        "num_failure": int(success.size - np.sum(success)),
        "episode_length": _quantile_block(lengths),
        "first_success_step": _quantile_block(solved) if solved.size else {},
        "bytes_total": total_bytes,
        "bytes_per_episode": total_bytes / max(n_episodes, 1),
        "bytes_per_channel": per_channel,
        "bytes_per_channel_per_episode": {
            name: value / max(n_episodes, 1) for name, value in per_channel.items()
        },
    }
    if project_to:
        report["projection"] = {
            "episodes": project_to,
            "bytes": total_bytes / max(n_episodes, 1) * project_to,
            "gib": total_bytes / max(n_episodes, 1) * project_to / (1024**3),
        }
    return report


def contact_stats(store_path: str) -> Dict[str, object]:
    """Per contact channel: how often it is live, and its peak force.

    A channel that is live 0% of the time is a real fact about the task, not a
    collector bug -- in peg insertion no robot body ever reaches the socket, so
    the hand-env pairs are identically zero. Recording the number in the sidecar
    is what stops someone quoting the resulting exact-zero MI as a measurement.
    """
    buffer = ReplayBuffer.create_from_path(store_path, mode="r")
    stats: Dict[str, object] = {}
    for name in sorted(k for k in buffer.keys() if k.startswith("contact")):
        arr = np.asarray(buffer[name][:], dtype=np.float64)
        force = np.linalg.norm(arr[:, 1:], axis=1)
        stats[name] = {
            "live_fraction": float(arr[:, 0].mean()),
            "force_max": float(force.max()),
            "force_mean_when_live": (
                float(force[arr[:, 0] > 0].mean()) if arr[:, 0].any() else 0.0
            ),
            "finite": bool(np.isfinite(arr).all()),
            "binary_in_contact": bool(set(np.unique(arr[:, 0]).tolist()) <= {0.0, 1.0}),
        }
    return stats


def censoring(store_path: str, horizons: Sequence[int]) -> Dict[str, object]:
    """How many pilot successes each candidate ``max_steps`` would censor."""
    buffer = ReplayBuffer.create_from_path(store_path, mode="r")
    ends = np.asarray(buffer.episode_ends[:], dtype=np.int64)
    n_episodes = int(np.diff(np.concatenate([[0], ends])).size)
    first = _meta(buffer, "first_success_step", n_episodes, np.int64, fill=-1)
    solved = first[first >= 0]
    return {
        str(h): {
            "kept": int(np.sum(solved < h)),
            "censored": int(np.sum(solved >= h)),
            "of_successes": int(solved.size),
        }
        for h in horizons
    }


def _meta(
    buffer: ReplayBuffer, key: str, size: int, dtype, fill: int = 0
) -> np.ndarray:
    if key in buffer.meta:
        return np.asarray(buffer.meta[key][:], dtype=dtype)
    return np.full(size, fill, dtype=dtype)


def _quantile_block(values: np.ndarray) -> Dict[str, float]:
    block = {f"p{q}": float(np.percentile(values, q)) for q in QUANTILES}
    block["mean"] = float(np.mean(values))
    return block


def format_report(report: Dict[str, object], label: str = "") -> str:
    lines: List[str] = []
    head = f"PILOT REPORT {label}".strip()
    lines.append("=" * 78)
    lines.append(head)
    lines.append(f"  store={report['store']}")
    lines.append(
        f"  episodes={report['num_episodes']}  steps={report['num_steps']}  "
        f"success={report['num_success']} "
        f"({100.0 * float(report['success_rate']):.1f}%)  "
        f"failure={report['num_failure']}"
    )
    lengths = report["episode_length"]
    lines.append(
        "  episode_length  "
        + "  ".join(f"{k}={v:.0f}" for k, v in sorted(lengths.items()))
    )
    if report["first_success_step"]:
        first = report["first_success_step"]
        lines.append(
            "  first_success   "
            + "  ".join(f"{k}={v:.0f}" for k, v in sorted(first.items()))
        )
    gib = float(report["bytes_total"]) / (1024**3)
    per_ep = float(report["bytes_per_episode"]) / (1024**2)
    lines.append(f"  size {gib:.2f} GiB total, {per_ep:.1f} MiB/episode")
    for name, value in sorted(
        report["bytes_per_channel_per_episode"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"    {name:<32} {value / (1024 ** 2):8.2f} MiB/episode")
    if "projection" in report:
        proj = report["projection"]
        lines.append(
            f"  projected to {proj['episodes']} episodes: {proj['gib']:.1f} GiB"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def write_report(report: Dict[str, object], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
