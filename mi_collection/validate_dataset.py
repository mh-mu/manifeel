"""Structural validation of a finished MI dataset, before publishing it.

    python -m mi_collection.validate_dataset data/mi_data/manifeel_pih_*_2000.zarr

Checks the invariants the fbsource readers depend on: every ``meta`` array has
one entry per episode, every ``data/*`` channel is as long as the last episode
end, the contact channels are ``(N, 4)`` finite with a binary column 0, the
truncation rule holds, and the sidecar carries what the analyzer needs
(``control_dt``, and all three ``contact_pairs`` labels).

Exits non-zero if anything fails, so it can gate the upload.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

from mi_collection.contact import CONTACT_STATE_DIM
from mi_collection.task_spec import CONTACT_LABELS

#: ``meta`` arrays the collector must write, one entry per episode.
REQUIRED_META: Tuple[str, ...] = (
    "episode_ends",
    "episode_success",
    "first_success_step",
    "episode_batch_seed",
    "episode_env_index",
)


def validate(path: str, expect_episodes: int | None = None) -> List[str]:
    """Return a list of failure messages; empty means the dataset is sound."""
    failures: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    buffer = ReplayBuffer.create_from_path(path, mode="r")
    ends = np.asarray(buffer.episode_ends[:], dtype=np.int64)
    n_episodes = int(ends.size)
    lengths = np.diff(np.concatenate([[0], ends]))

    if expect_episodes is not None:
        check(
            n_episodes == expect_episodes,
            f"expected {expect_episodes} episodes, found {n_episodes}",
        )
    for key in REQUIRED_META:
        if key not in buffer.meta:
            failures.append(f"meta/{key} is missing")
            continue
        size = int(np.asarray(buffer.meta[key][:]).size)
        check(
            size == n_episodes,
            f"meta/{key} has {size} entries, expected {n_episodes}",
        )

    for key in sorted(buffer.keys()):
        rows = int(buffer[key].shape[0])
        check(
            rows == int(ends[-1]),
            f"data/{key} has {rows} rows, expected {int(ends[-1])}",
        )

    info = _sidecar(path)
    check(info.get("control_dt") is not None, "sidecar has no control_dt")
    pairs = info.get("contact_pairs") or {}
    check(bool(pairs), "sidecar has no contact_pairs map")
    # A subset, not equality: a task need not produce every pair type, and the
    # hand-env channels are deliberately dropped from the peg-insertion datasets
    # (no robot body reaches the socket). The analyzer skips an absent group.
    check(
        set(pairs.values()) <= set(CONTACT_LABELS),
        f"sidecar contact_pairs has unknown labels "
        f"{sorted(set(pairs.values()) - set(CONTACT_LABELS))}",
    )
    for name in pairs:
        check(name in buffer.keys(), f"sidecar names {name}, which is not a channel")

    for name in sorted(k for k in buffer.keys() if k.startswith("contact")):
        arr = np.asarray(buffer[name][:])
        check(
            arr.ndim == 2 and arr.shape[1] == CONTACT_STATE_DIM,
            f"{name} has shape {arr.shape}, expected (N, {CONTACT_STATE_DIM})",
        )
        check(np.isfinite(arr).all(), f"{name} has non-finite values")
        check(
            set(np.unique(arr[:, 0]).tolist()) <= {0.0, 1.0},
            f"{name} column 0 is not binary",
        )

    if "motion_primitive" in buffer.keys():
        vocabulary = (info.get("motion_primitives") or {}).get("vocabulary") or []
        check(bool(vocabulary), "motion_primitive present but sidecar has no vocabulary")
        codes = np.asarray(buffer["motion_primitive"][:])
        check(
            int(codes.max(initial=0)) < len(vocabulary),
            f"motion_primitive has codes >= the {len(vocabulary)}-entry vocabulary",
        )
        phases = np.asarray(buffer["primitive_phase"][:])
        check(
            bool(np.all((phases >= 0.0) & (phases <= 1.0))),
            "primitive_phase is outside [0, 1]",
        )

    success = np.asarray(buffer.meta["episode_success"][:], dtype=np.int64)
    first = np.asarray(buffer.meta["first_success_step"][:], dtype=np.int64)
    max_steps = int(info.get("max_steps", lengths.max()))
    tail = int(info.get("success_tail_steps", 20))
    expected = np.where(
        success > 0, np.minimum(first + 1 + tail, max_steps), max_steps
    )
    bad = int(np.count_nonzero(expected != lengths))
    check(bad == 0, f"{bad} episodes violate the truncation rule")
    check(
        bool(np.all((first >= 0) == (success > 0))),
        "first_success_step and episode_success disagree",
    )
    check(int(success.sum()) > 0, "no successful episodes -- no success subset")
    check(int((success == 0).sum()) > 0, "no failed episodes -- no failure subset")
    return failures


def summarize(path: str) -> Dict[str, object]:
    buffer = ReplayBuffer.create_from_path(path, mode="r")
    success = np.asarray(buffer.meta["episode_success"][:], dtype=np.int64)
    return {
        "episodes": int(np.asarray(buffer.episode_ends[:]).size),
        "steps": int(buffer.n_steps),
        "success": int(success.sum()),
        "failure": int((success == 0).sum()),
        "channels": len(list(buffer.keys())),
    }


def _sidecar(path: str) -> Dict[str, object]:
    info_path = os.path.join(path, "mi_collect_info.json")
    if not os.path.exists(info_path):
        return {}
    with open(info_path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    expect = None
    for flag in sys.argv[1:]:
        if flag.startswith("--expect-episodes="):
            expect = int(flag.split("=", 1)[1])
    if not args:
        print(__doc__)
        sys.exit(2)
    bad = 0
    for path in args:
        path = path.rstrip("/")
        failures = validate(path, expect)
        stats = summarize(path)
        status = "OK  " if not failures else "FAIL"
        print(
            f"[{status}] {os.path.basename(path)}  "
            f"episodes={stats['episodes']} steps={stats['steps']} "
            f"success={stats['success']} failure={stats['failure']} "
            f"channels={stats['channels']}"
        )
        for failure in failures:
            print(f"         - {failure}")
        bad += bool(failures)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
