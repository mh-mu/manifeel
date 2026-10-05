"""Live smoke test: the claims that can only be checked against a real sim.

Not part of ``unittest discover`` (it needs a GPU, a checkpoint and ~2 min).

    MANIFEEL_GPU=0 python tests/mi_collection/live_smoke.py \
        -c data/outputs/vistac_pih_50/44/checkpoints/latest_epoch50.ckpt

Checks, in the order the plan lists them:

1. the GPU pipeline is on and ``contact_force_pairwise`` is not ``None``
2. every ``shape_meta`` key is present in ``render_cache``
3. image ranges and CHW layout are what the writer assumes
4. exactly one stored row per simulator step
5. no reset after success (``progress_buf`` never rewinds)
6. the stored contact matches an independent recomputation on sampled frames
7. an episode is ``min(first_success_step + 1 + tail, max_steps)`` long when it
   succeeded and exactly ``max_steps`` when it did not
"""

import isaacgym  # noqa: F401  # must precede torch, always

import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import argparse  # noqa: E402
import pathlib  # noqa: E402
import tempfile  # noqa: E402

import dill  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402

from mi_collection.collector import CollectOptions, MICollector  # noqa: E402
from mi_collection.task_specs import get_task_spec  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)

FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not condition:
        FAILURES.append(name)


def numpy_pair_force(pairwise, ids_a, ids_b):
    """Independent recomputation of the descriptor, with explicit loops."""
    force = np.zeros(3, dtype=np.float64)
    live = False
    for i in ids_a:
        for j in ids_b:
            entry = pairwise[i, j].astype(np.float64)
            force += entry
            live = live or float(np.linalg.norm(entry)) > 1e-3
    return live, force


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--tail", type=int, default=20)
    parser.add_argument("--task-spec", default="peg_insertion")
    args = parser.parse_args()

    hydra.initialize_config_dir(
        config_dir=str(
            pathlib.Path(__file__).resolve().parents[2].joinpath("manifeel", "config")
        ),
        version_base=None,
    )
    work = tempfile.mkdtemp(prefix="mi_live_smoke_")
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=work)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(args.device))
    policy.eval()

    spec = get_task_spec(args.task_spec)
    options = CollectOptions(
        out_dir=work,
        num_episodes=args.num_envs,
        num_envs=args.num_envs,
        start_seed=777000,
        max_steps=args.max_steps,
        success_tail_steps=args.tail,
        keep_shards=True,
    )
    collector = MICollector(cfg, policy, torch.device(args.device), spec, options)
    collector.build()

    task = collector.inner.envs
    print("\n=== live checks ===")
    check(
        "1. gpu pipeline exposes contact_force_pairwise",
        task.contact_force_pairwise is not None,
        str(tuple(task.contact_force_pairwise.shape)),
    )

    # Instrument the recorder: keep the raw pairwise tensor and progress_buf of
    # every step, so the stored rows can be checked against the sim itself.
    raw = {"pairwise": [], "progress": [], "cache_dtypes": {}}
    original = collector.recorder._snapshot_pre_step

    def instrumented():
        row = original()
        raw["pairwise"].append(
            task.contact_force_pairwise.detach().to("cpu").numpy().copy()
        )
        raw["progress"].append(task.progress_buf.detach().to("cpu").numpy().copy())
        if not raw["cache_dtypes"]:
            raw["cache_dtypes"] = {
                key: (value.dtype, value.shape)
                for key, value in collector.inner.render_cache.items()
            }
        return row

    collector.recorder._snapshot_pre_step = instrumented

    result = collector.run()
    shard = ReplayBuffer.create_from_path(result["shard_paths"][0], mode="r")

    shape_meta_keys = [
        k for k in cfg.task.shape_meta["obs"].keys() if k not in ("state",)
    ]
    check(
        "2. every shape_meta key is in render_cache",
        all(k in raw["cache_dtypes"] for k in shape_meta_keys),
        f"{shape_meta_keys}",
    )

    wrist = shard["wrist"]
    taxim = shard["right_tactile_camera_taxim"]
    tacff = shard["tactile_force_field_right"]
    check(
        "3a. wrist is uint8 CHW 3x256x256",
        wrist.dtype == np.uint8 and wrist.shape[1:] == (3, 256, 256),
        f"{wrist.dtype} {wrist.shape}",
    )
    check(
        "3b. wrist spans a real range (not a constant frame)",
        int(np.asarray(wrist[0]).max()) > int(np.asarray(wrist[0]).min()) + 16,
        f"min={np.asarray(wrist[0]).min()} max={np.asarray(wrist[0]).max()}",
    )
    check(
        "3c. taxim is uint8 CHW 3x320x240",
        taxim.dtype == np.uint8 and taxim.shape[1:] == (3, 320, 240),
        f"{taxim.dtype} {taxim.shape}",
    )
    check(
        "3d. tactile force field is float32 (10, 14, 3)",
        tacff.dtype == np.float32 and tacff.shape[1:] == (10, 14, 3),
        f"{tacff.dtype} {tacff.shape}",
    )

    lengths = np.diff(np.concatenate([[0], np.asarray(shard.episode_ends[:])]))
    dt = collector.control_dt
    gaps = []
    for slot in range(shard.n_episodes):
        stamps = np.asarray(shard.get_episode(slot)["timestamp"]).reshape(-1)
        if not np.allclose(stamps, np.arange(stamps.size) * dt, atol=1e-6):
            gaps.append(slot)
    # The driver steps in n_action_steps-sized chunks, so it can overshoot the
    # cap by up to n_action_steps-1; those rows are pushed to an accumulator
    # whose envs are all complete and are discarded. The invariant is that the
    # longest episode reaches the cap and no episode has a gap in its clock.
    driven = len(raw["progress"])
    n_action_steps = int(cfg.n_action_steps)
    check(
        "4. one row per simulator step (timestamps are a gapless t*control_dt)",
        not gaps
        and int(lengths.max()) <= driven < int(lengths.max()) + n_action_steps,
        f"driven={driven} lengths={list(lengths)} dt={dt:.6f} gaps={gaps} "
        f"n_action_steps={n_action_steps}",
    )

    progress = np.stack(raw["progress"], axis=0)
    check(
        "5. no reset after success (progress_buf never rewinds)",
        bool(np.all(np.diff(progress, axis=0) >= 0)),
        f"progress {progress[0].tolist()} -> {progress[-1].tolist()}",
    )

    pairs = collector.pair_indices
    env_index = np.asarray(shard.meta["episode_env_index"][:])
    order = np.asarray(shard.meta["episode_write_order"][:])
    mismatch = 0
    sampled = 0
    for slot, env in zip(order, env_index):
        episode = shard.get_episode(int(slot))
        for step in range(0, len(episode["timestamp"]), 17):
            frame = raw["pairwise"][step][int(env)]
            for pair in pairs:
                live, force = numpy_pair_force(frame, pair.ids_a, pair.ids_b)
                stored = np.asarray(episode[pair.name][step], dtype=np.float64)
                sampled += 1
                if not np.allclose(stored[1:], force, atol=1e-5) or bool(
                    stored[0]
                ) != live:
                    mismatch += 1
    check(
        "6. stored contact matches an independent recomputation",
        mismatch == 0,
        f"{sampled - mismatch}/{sampled} sampled (pair, frame) values agree",
    )

    success = np.asarray(shard.meta["episode_success"][:])
    first = np.asarray(shard.meta["first_success_step"][:])
    # `order[k]` is the storage slot of the k-th env, and the meta arrays are in
    # env order -- so this puts the lengths in env order too.
    stored_lengths = lengths[order]
    expected = np.where(
        success > 0,
        np.minimum(first + 1 + args.tail, args.max_steps),
        args.max_steps,
    )
    check(
        "7. episode lengths follow the truncation rule",
        bool(np.all(stored_lengths == expected)),
        f"success={success.tolist()} first={first.tolist()} "
        f"len={stored_lengths.tolist()} expected={expected.tolist()}",
    )
    if not success.any():
        print(
            "  [NOTE] no success in this smoke run -- rule 7 only exercised its "
            "failure branch. Re-run with a longer --max-steps to cover both."
        )

    print(f"\n{len(FAILURES)} failed check(s): {FAILURES}" if FAILURES else "\nALL OK")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
