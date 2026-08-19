"""Drop channels from a finished MI dataset, in place, with the sidecar.

    python -m mi_collection.drop_channels --channels contact_left_socket \
        contact_right_socket -- data/mi_data/manifeel_pih_*_2000.zarr

Used to remove the hand-env contact pairs from the peg-insertion datasets: no
robot body ever reaches the socket in this task, so ``contact_left_socket`` and
``contact_right_socket`` are identically zero for every step of every episode.
They cost nothing in storage but they are eight constant columns inside
``contacts_all``, and side dimensionality is one of the levers that moves the DV
cliff (see fbsource `.llms/rules/mine_dv_epoch_collapse.md`) -- so carrying them
makes the pooled target strictly worse to estimate.

Deleting a zarr array is a directory removal, so this is cheap: the remaining
channels, ``episode_ends`` and every other ``meta`` array are untouched. The
sidecar's ``contact_pairs`` / ``contact_stats`` / ``roles`` are pruned to match
and a ``dropped_channels`` block records what went and why.

⚠️ The fbsource analyzer groups contact channels by the sidecar's
``contact_pairs`` labels. Removing every channel of a label removes that group;
``analyze_contact_pairs`` skips the group with a warning rather than failing,
but only from the 2026-08-19 change onwards.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Dict, List, Sequence

import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer

_SIDECAR = "mi_collect_info.json"

#: Sidecar maps keyed by channel name, pruned alongside the arrays.
_CHANNEL_KEYED = ("contact_pairs", "contact_stats", "contact_pair_bodies", "roles")


def drop(
    path: str, channels: Sequence[str], reason: str, dry_run: bool = False
) -> Dict[str, object]:
    """Remove ``channels`` from the store at ``path``.  Returns what changed."""
    path = os.path.expanduser(path.rstrip("/"))
    root = zarr.open(path, mode="r" if dry_run else "a")
    present = [name for name in channels if name in root["data"]]
    absent = [name for name in channels if name not in root["data"]]
    remaining = [name for name in root["data"] if name not in present]
    if not remaining:
        raise ValueError(f"{path}: dropping {list(channels)} would empty the store.")
    freed = sum(
        _directory_bytes(os.path.join(path, "data", name)) for name in present
    )
    if dry_run:
        return {
            "path": path,
            "dropped": present,
            "absent": absent,
            "remaining": sorted(remaining),
            "bytes_freed": freed,
            "dry_run": True,
        }

    for name in present:
        del root["data"][name]
        # zarr leaves the (now empty) directory behind on some stores.
        stale = os.path.join(path, "data", name)
        if os.path.isdir(stale):
            shutil.rmtree(stale)

    info = _prune_sidecar(path, present, reason)
    verify = _verify(path, present)
    return {
        "path": path,
        "dropped": present,
        "absent": absent,
        "remaining": sorted(verify["channels"]),
        "bytes_freed": freed,
        "episodes": verify["episodes"],
        "steps": verify["steps"],
        "sidecar_labels": sorted(set((info.get("contact_pairs") or {}).values())),
    }


def _prune_sidecar(path: str, dropped: List[str], reason: str) -> Dict[str, object]:
    sidecar = os.path.join(path, _SIDECAR)
    if not os.path.exists(sidecar):
        return {}
    with open(sidecar, encoding="utf-8") as handle:
        info = json.load(handle)
    for key in _CHANNEL_KEYED:
        block = info.get(key)
        if isinstance(block, dict):
            info[key] = {k: v for k, v in block.items() if k not in dropped}
    never_live = info.get("contact_channels_never_live")
    if isinstance(never_live, list):
        info["contact_channels_never_live"] = [
            n for n in never_live if n not in dropped
        ]
    history = info.setdefault("dropped_channels", [])
    history.append({"channels": list(dropped), "reason": reason})
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)
    return info


def _verify(path: str, dropped: Sequence[str]) -> Dict[str, object]:
    """Reopen through ReplayBuffer: the store must still be internally consistent."""
    buffer = ReplayBuffer.create_from_path(path, mode="r")
    channels = list(buffer.keys())
    leftover = [name for name in dropped if name in channels]
    if leftover:
        raise RuntimeError(f"{path}: {leftover} survived the drop.")
    ends = buffer.episode_ends[:]
    for name in channels:
        if buffer[name].shape[0] != int(ends[-1]):
            raise RuntimeError(
                f"{path}: data/{name} has {buffer[name].shape[0]} rows, "
                f"expected {int(ends[-1])}."
            )
    return {"channels": channels, "episodes": int(len(ends)), "steps": int(ends[-1])}


def _directory_bytes(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, files in os.walk(path)
        for name in files
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", nargs="+", required=True)
    parser.add_argument(
        "--reason",
        default="identically zero in this task -- no robot body reaches the socket",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("stores", nargs="+")
    args = parser.parse_args()

    for store in args.stores:
        result = drop(store, args.channels, args.reason, args.dry_run)
        prefix = "[dry-run] " if args.dry_run else ""
        print(
            f"{prefix}{os.path.basename(result['path'])}: dropped "
            f"{result['dropped']}"
            + (f", absent {result['absent']}" if result["absent"] else "")
            + f", freed {result['bytes_freed'] / 1024 ** 2:.1f} MiB"
        )
        print(f"    remaining channels ({len(result['remaining'])}): "
              f"{result['remaining']}")
        if not args.dry_run:
            print(
                f"    verified: {result['episodes']} episodes, "
                f"{result['steps']} steps, labels {result['sidecar_labels']}"
            )


if __name__ == "__main__":
    main()
