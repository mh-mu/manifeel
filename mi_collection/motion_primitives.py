"""Per-step motion-primitive labels for object search, and their offline relabeller.

Object search is not time-synced: one episode finds the battery in 5 s, the next
searches for 40 s and regrasps twice.  MI read per timestep therefore averages
unlike motions together; MI read per *primitive* does not.  Every step gets one
label from :data:`PRIMITIVES`, plus ``primitive_phase`` -- progress 0 -> 1 through
the contiguous run of that label it sits in -- so runs of different lengths can be
time-warped onto each other.

Labels are a pure function of channels the collector already records
(``state_full``, ``action``, the ``contact_*`` descriptors), computed on the whole
finished episode, so velocities can be smoothed non-causally.  The same function
runs offline: thresholds can be re-tuned and a published store relabelled without
a simulator::

    python -m mi_collection.motion_primitives relabel <store.zarr> [--param k=v ...]
    python -m mi_collection.motion_primitives summary <store.zarr>

Besides the dense ``data/motion_primitive`` / ``data/primitive_phase`` channels,
each primitive's first entry per episode is written to ``meta/stage_<primitive>``
in the contract of fbsource ``record_manifeel_stages.py`` (global flat index one
step *before* the first in-primitive observation; the reader adds it back), so the
analyzers' ``--stage-order objsearch`` / ``--align stage`` read it unmodified.
Primitives recur within an episode; ``meta/stage_*`` keeps only the first entry,
the dense channel keeps every one.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import zarr

#: Label vocabulary; the uint8 code is the index.  Order is the nominal task order.
PRIMITIVES: Tuple[str, ...] = (
    "descend",  # gripper open, moving down, touching nothing
    "search",  # gripper open, low in the box, touching nothing
    "probe",  # gripper open, a finger touching the item or a box wall
    "close",  # gripper closing, or commanded closed in the box / on the item
    "lift",  # commanded closed, item held by both fingers, lifted and rising
    "hold",  # commanded closed, item held and lifted, not rising
    "failed_lift",  # commanded closed, item not held, hand rising or above the rim
    "release",  # gripper opening
    "hover",  # gripper open, above the rim, not descending, touching nothing
)
CODES: Dict[str, int] = {name: i for i, name in enumerate(PRIMITIVES)}
VERSION: int = 2

#: ``state_full`` columns (object-search layout, see ``ObjectSearchSpec``).
_EE_Z, _ITEM_Z, _BOX_Z = 2, 9, 16
_GRIPPER = slice(21, 23)
_GRIPPER_CMD = 6  # action column


@dataclasses.dataclass
class PrimitiveParams:
    """Thresholds; recorded in the sidecar so a label can be traced to them."""

    #: action[6] below this is a close command (0.040 open, ~0.020 closed in demos).
    #: The policies also narrow the gripper part-way (0.030-0.035) and reopen;
    #: ``width_rate`` catches those as close / release.
    close_command: float = 0.030
    #: EE height above the box origin below which the hand is in the box (starts at
    #: 0.074, searches at 0.007-0.04 in the pilot; vistac hovers at 0.05-0.08 when
    #: it stalls late in a failed episode).
    rim_height: float = 0.05
    #: Item centre height above the box origin when it rests upright.
    item_rest_height: float = 0.0048
    #: Item height above rest that counts as lifted.
    lifted: float = 0.02
    #: Vertical speed (m/s) that counts as moving up / down.
    vz_moving: float = 0.02
    #: Gripper-width rate (m/s, both fingers summed) that counts as closing/opening.
    width_rate: float = 0.02
    #: Centered moving-average window (steps) for every finite difference.
    smooth: int = 5
    #: Runs shorter than this many steps are merged into the preceding run.
    min_run: int = 3

    def to_dict(self) -> Dict[str, float]:
        return dataclasses.asdict(self)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < 2:
        return x
    pad = window // 2
    padded = np.pad(x, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def _rate(x: np.ndarray, dt: float, window: int) -> np.ndarray:
    if x.size < 2:
        return np.zeros_like(x)
    return _smooth(np.gradient(x, dt), window)


def _in_contact(channels: Dict[str, np.ndarray], name: str, n: int) -> np.ndarray:
    if name not in channels:
        return np.zeros(n, dtype=bool)
    return np.asarray(channels[name])[:, 0] > 0.5


def raw_labels(
    channels: Dict[str, np.ndarray], control_dt: float, params: PrimitiveParams
) -> np.ndarray:
    """Per-step codes from the priority rules, before run merging."""
    state = np.asarray(channels["state_full"], dtype=np.float64)
    n = state.shape[0]
    box_z = state[:, _BOX_Z]
    ee_h = state[:, _EE_Z] - box_z
    item_h = state[:, _ITEM_Z] - box_z - params.item_rest_height
    width = state[:, _GRIPPER].sum(axis=1)
    closed = np.asarray(channels["action"])[:, _GRIPPER_CMD] < params.close_command

    vz_ee = _rate(ee_h, control_dt, params.smooth)
    vz_item = _rate(item_h, control_dt, params.smooth)
    v_width = _rate(width, control_dt, params.smooth)

    left = _in_contact(channels, "contact_left_item", n)
    right = _in_contact(channels, "contact_right_item", n)
    held = left & right
    touching = left | right
    for name in ("contact_left_box", "contact_right_box"):
        touching |= _in_contact(channels, name, n)
    lifted = item_h >= params.lifted

    in_box = ee_h <= params.rim_height
    closing = v_width < -params.width_rate
    labels = np.full(n, CODES["hover"], dtype=np.uint8)
    # Lowest priority first: every later assignment wins where it applies.
    labels[in_box] = CODES["search"]
    labels[~touching & (vz_ee < -params.vz_moving)] = CODES["descend"]
    labels[touching] = CODES["probe"]
    labels[v_width > params.width_rate] = CODES["release"]
    labels[closing | (closed & (in_box | touching))] = CODES["close"]
    # The vision / tacff policies often close on nothing and carry the empty
    # gripper far above the box (pilot: 0.2-0.7 m): that is the failed lift too.
    empty_up = (vz_ee > params.vz_moving) | ~in_box
    labels[closed & ~held & empty_up] = CODES["failed_lift"]
    labels[closed & held & lifted] = CODES["hold"]
    labels[closed & held & lifted & (vz_item > params.vz_moving)] = CODES["lift"]
    return labels


def merge_short_runs(labels: np.ndarray, min_run: int) -> np.ndarray:
    """Absorb runs shorter than ``min_run`` into the run before them.

    A leading short run is absorbed into the run after it instead.  Removes
    single-step flicker at threshold crossings without moving real boundaries by
    more than ``min_run - 1`` steps.
    """
    out = labels.copy()
    if min_run <= 1 or out.size == 0:
        return out
    starts, lengths = run_bounds(out)
    for start, length in zip(starts, lengths):
        if length >= min_run:
            continue
        if start > 0:
            out[start : start + length] = out[start - 1]
        elif start + length < out.size:
            out[start : start + length] = out[start + length]
    return out


def run_bounds(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(starts, lengths)`` of the contiguous runs of equal labels."""
    if labels.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(np.diff(labels.astype(np.int64))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [labels.size]])
    return starts, ends - starts


def phase(labels: np.ndarray) -> np.ndarray:
    """Progress 0 -> 1 through each contiguous run (0 for a one-step run)."""
    out = np.zeros(labels.size, dtype=np.float32)
    starts, lengths = run_bounds(labels)
    for start, length in zip(starts, lengths):
        if length > 1:
            out[start : start + length] = np.arange(length) / (length - 1)
    return out


def label_episode(
    channels: Dict[str, np.ndarray],
    control_dt: float,
    params: PrimitiveParams | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(labels uint8 (T,), phase float32 (T,))`` for one episode."""
    params = params or PrimitiveParams()
    labels = merge_short_runs(raw_labels(channels, control_dt, params), params.min_run)
    return labels, phase(labels)


def first_entries(
    labels: np.ndarray,
    episode_ends: np.ndarray,
    first_success: np.ndarray,
) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
    """``({stage: [stored global index]}, {stage: episodes entering at step 0})``.

    The stored index is one before the first in-stage step (fbsource
    ``record_manifeel_stages.py`` contract); an entry at an episode's step 0 has
    no representable index and is counted instead.  ``success`` comes from
    ``meta/first_success_step``.
    """
    stages = list(PRIMITIVES) + ["success"]
    entries: Dict[str, List[int]] = {name: [] for name in stages}
    dropped: Dict[str, int] = {name: 0 for name in stages}
    starts = np.concatenate([[0], episode_ends[:-1]])
    for episode, (start, end) in enumerate(zip(starts, episode_ends)):
        segment = labels[start:end]
        local_first: Dict[str, int] = {}
        for code in np.unique(segment):
            local_first[PRIMITIVES[int(code)]] = int(np.argmax(segment == code))
        if first_success[episode] >= 0:
            local_first["success"] = int(first_success[episode])
        for name, local in local_first.items():
            if local == 0:
                dropped[name] += 1
            else:
                entries[name].append(int(start + local - 1))
    return entries, dropped


def write_store_annotations(
    store: str, params: PrimitiveParams | None = None, relabel: bool = False
) -> Dict[str, object]:
    """Write ``meta/stage_*`` (and with ``relabel``, the dense channels) into a store.

    Returns the sidecar block, which is also merged into ``mi_collect_info.json``.
    """
    params = params or PrimitiveParams()
    root = zarr.open(os.path.expanduser(store), mode="a")
    data, meta = root["data"], root["meta"]
    ends = np.asarray(meta["episode_ends"][:], dtype=np.int64)
    if relabel:
        control_dt = float(_read_sidecar(store)["control_dt"])
        _relabel_dense(data, ends, control_dt, params)
    labels = np.asarray(data["motion_primitive"][:]).reshape(-1)
    first_success = np.asarray(meta["first_success_step"][:], dtype=np.int64)
    entries, dropped = first_entries(labels, ends, first_success)
    for name in list(meta.array_keys()):
        if name.startswith("stage_"):
            del meta[name]
    for name, indices in entries.items():
        if indices:
            meta.array(
                f"stage_{name}",
                np.asarray(indices, dtype="<i8"),
                chunks=(len(indices),),
                compressor=None,
            )
    block = {
        "version": VERSION,
        "vocabulary": list(PRIMITIVES),
        "params": params.to_dict(),
        "channels": {
            "motion_primitive": "uint8 code, index into vocabulary",
            "primitive_phase": "float32, 0 -> 1 through the current run",
        },
        "description": (
            "meta/stage_<name>: global flat step index (meta/episode_ends "
            "coordinates) one step before each episode's FIRST step labelled "
            "<name>; the fbsource stage reader adds the step back. Primitives "
            "recur; only the dense channel has every entry."
        ),
        "meta_arrays": {f"stage_{k}": len(v) for k, v in entries.items() if v},
        "episodes_entering_at_step_zero": {k: v for k, v in dropped.items() if v},
        "step_fraction": step_fractions(labels),
    }
    _merge_sidecar(store, {"motion_primitives": block})
    return block


def _relabel_dense(data, ends, control_dt, params) -> None:
    keys = ["state_full", "action"] + [k for k in data.array_keys() if k.startswith("contact_")]
    arrays = {k: np.asarray(data[k][:]) for k in keys}
    labels = np.zeros(int(ends[-1]), dtype=np.uint8)
    phases = np.zeros(int(ends[-1]), dtype=np.float32)
    start = 0
    for end in ends:
        episode = {k: v[start:end] for k, v in arrays.items()}
        labels[start:end], phases[start:end] = label_episode(episode, control_dt, params)
        start = int(end)
    data["motion_primitive"][:] = labels.reshape(-1, 1)
    data["primitive_phase"][:] = phases.reshape(-1, 1)


def step_fractions(labels: np.ndarray) -> Dict[str, float]:
    counts = np.bincount(labels.astype(np.int64), minlength=len(PRIMITIVES))
    total = max(int(counts.sum()), 1)
    return {name: round(float(counts[i]) / total, 4) for i, name in enumerate(PRIMITIVES)}


def _read_sidecar(store: str) -> Dict[str, object]:
    path = os.path.join(os.path.expanduser(store), "mi_collect_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _merge_sidecar(store: str, update: Dict[str, object]) -> None:
    path = os.path.join(os.path.expanduser(store), "mi_collect_info.json")
    info = _read_sidecar(store)
    info.update(update)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, sort_keys=True)


def _parse_params(pairs: Sequence[str], base: PrimitiveParams) -> PrimitiveParams:
    fields = {f.name: f.type for f in dataclasses.fields(PrimitiveParams)}
    values = base.to_dict()
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if key not in fields:
            raise SystemExit(f"unknown param {key!r}; have {sorted(fields)}")
        values[key] = int(raw) if key in ("smooth", "min_run") else float(raw)
    return PrimitiveParams(**values)


def _summary(store: str) -> None:
    info = _read_sidecar(store).get("motion_primitives", {})
    print(json.dumps(info, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    relabel = sub.add_parser("relabel", help="recompute labels from stored channels")
    relabel.add_argument("store")
    relabel.add_argument("--param", action="append", default=[], help="k=v override")
    summary = sub.add_parser("summary", help="print the sidecar block")
    summary.add_argument("store")
    args = parser.parse_args(argv)
    if args.command == "summary":
        _summary(args.store)
        return
    stored = _read_sidecar(args.store).get("motion_primitives", {}).get("params", {})
    # Start from the params the store was labelled with; keys a newer version
    # renamed or dropped fall back to the defaults.
    known = {f.name for f in dataclasses.fields(PrimitiveParams)}
    base = PrimitiveParams(**{k: v for k, v in stored.items() if k in known})
    block = write_store_annotations(
        args.store, _parse_params(args.param, base), relabel=True
    )
    print(json.dumps({k: block[k] for k in ("params", "meta_arrays", "step_fraction")},
                     indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
