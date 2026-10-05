"""Collect a mutual-information rollout dataset from a trained ManiFeel policy.

Mirrors ``eval.py`` for everything to do with the checkpoint (isaacgym first,
the ``eval`` resolver, ``initialize_config_dir``, ``torch.load(..., dill)``, the
workspace + ``load_payload``, EMA selection), then rolls the policy out and
records every modality per simulator step alongside a ground-truth contact
descriptor and per-episode success labels.

Usage:

    MANIFEEL_GPU=0 python collect_mi.py \
        --checkpoint data/outputs/vistac_pih_50/44/checkpoints/latest_epoch50.ckpt \
        --out data/mi_data/manifeel_pih_vistac_1000 \
        --num-episodes 1000 --num-envs 50 --max-steps 250 -d cuda:0
"""

import isaacgym  # noqa: F401  # must precede torch, always

import sys

# line-buffer both streams, as eval.py does, so a tee'd log is live
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json  # noqa: E402
import os  # noqa: E402
import pathlib  # noqa: E402
import time  # noqa: E402

import click  # noqa: E402
import dill  # noqa: E402
import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa: E402

from mi_collection import pilot_report, writer as writer_mod  # noqa: E402
from mi_collection.collector import CollectOptions, MICollector  # noqa: E402
from mi_collection.contact import CONTACT_STATE_DIM  # noqa: E402
from mi_collection.recording_wrapper import DEFAULT_OBS_CHANNELS  # noqa: E402
from mi_collection.task_specs import get_task_spec  # noqa: E402

# the checkpoint's cfg may still carry unresolved ${eval:'...'} interpolations
OmegaConf.register_new_resolver("eval", eval, replace=True)

#: Candidate horizons the pilot reports censoring for.
_PILOT_HORIZONS = (100, 150, 200, 250, 300, 400, 500)


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--out", required=True, help="Output dataset directory (.zarr).")
@click.option("-d", "--device", default="cuda:0")
@click.option("--num-episodes", default=2000, type=int)
@click.option("--num-envs", default=50, type=int, help="Episodes per batch.")
@click.option("--start-seed", default=300000, type=int)
@click.option(
    "--max-steps",
    default=500,
    type=int,
    help="Hard cap on every episode. Also the search horizon for the first "
    "success: an episode with no success in this many steps is stored at "
    "exactly this length.",
)
@click.option(
    "--success-tail-steps",
    default=20,
    type=int,
    help="Steps recorded after the first success. An episode ends at the "
    "success tail or at --max-steps, whichever comes first.",
)
@click.option("--contact-threshold", default=1e-3, type=float)
@click.option("--task-spec", "task_spec_name", default="peg_insertion")
@click.option(
    "--include-plug-table/--no-include-plug-table",
    default=True,
    help="Record the task spec's optional contact pairs (peg insertion: "
    "plug<->table; object search: hand<->box and item<->table).",
)
@click.option(
    "--pilot",
    is_flag=True,
    help="Pilot mode: write the pilot report and the horizon-censoring table.",
)
@click.option("--resume", is_flag=True, help="Reuse complete, matching shards.")
@click.option("--keep-shards", is_flag=True, help="Do not delete shards after merge.")
@click.option("--overwrite", is_flag=True, help="Replace an existing output store.")
@click.option(
    "-x",
    "--override",
    multiple=True,
    help="dotlist override applied to the checkpoint cfg.",
)
def main(
    checkpoint,
    out,
    device,
    num_episodes,
    num_envs,
    start_seed,
    max_steps,
    success_tail_steps,
    contact_threshold,
    task_spec_name,
    include_plug_table,
    pilot,
    resume,
    keep_shards,
    overwrite,
    override,
):
    out = os.path.expanduser(out)
    work_dir = out + ".work"
    pathlib.Path(work_dir).mkdir(parents=True, exist_ok=True)

    policy, cfg = _load_policy(checkpoint, device, override, work_dir)
    task_spec = get_task_spec(task_spec_name)
    options = CollectOptions(
        out_dir=work_dir,
        num_episodes=num_episodes,
        num_envs=num_envs,
        start_seed=start_seed,
        max_steps=max_steps,
        success_tail_steps=success_tail_steps,
        contact_threshold=contact_threshold,
        include_optional_pairs=include_plug_table,
        obs_channels=DEFAULT_OBS_CHANNELS,
        resume=resume,
        keep_shards=keep_shards,
    )
    _print_header(checkpoint, out, cfg, options, task_spec)

    collector = MICollector(cfg, policy, torch.device(device), task_spec, options)
    started = time.time()
    result = collector.run()

    labels, provenance = result["contact_pairs"]
    merged = writer_mod.consolidate(
        result["shard_paths"], out, overwrite=overwrite, limit=num_episodes
    )
    stats = pilot_report.contact_stats(out)
    info = _sidecar(
        checkpoint=checkpoint,
        cfg=cfg,
        options=options,
        task_spec=task_spec,
        control_dt=result["control_dt"],
        labels=labels,
        provenance=provenance,
        merged=merged,
        batches=result["batches"],
        elapsed=time.time() - started,
        contact_stats=stats,
    )
    path = writer_mod.write_sidecar(out, info)
    print(f"\nWrote {merged['num_episodes']} episodes, {merged['num_steps']} steps")
    print(f"  success={merged['num_success']}  sidecar={path}")
    print("CONTACT CHANNELS")
    for name, row in stats.items():
        flag = "  <-- never live" if row["live_fraction"] == 0.0 else ""
        print(
            f"  {name:<24} {labels[name]:<12} live {row['live_fraction'] * 100:6.2f}%  "
            f"peak {row['force_max']:8.3f} N{flag}"
        )

    report = pilot_report.summarize(out, project_to=1000)
    print(pilot_report.format_report(report, label=os.path.basename(out)))
    pilot_report.write_report(report, os.path.join(out, "pilot_report.json"))
    if pilot:
        table = pilot_report.censoring(out, _PILOT_HORIZONS)
        print("HORIZON CENSORING (successes kept / censored)")
        for horizon, row in sorted(table.items(), key=lambda kv: int(kv[0])):
            print(
                f"  max_steps={horizon:>4}: kept {row['kept']:>4} / "
                f"censored {row['censored']:>4}  of {row['of_successes']} successes"
            )
        with open(
            os.path.join(out, "horizon_censoring.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(table, handle, indent=2, sort_keys=True)

    if not keep_shards:
        for shard in result["shard_paths"]:
            writer_mod.discard_shard(shard)
        _remove_if_empty(work_dir)


def _load_policy(checkpoint, device, override, output_dir):
    """eval.py's checkpoint path, verbatim in behaviour."""
    # ManifeelRunner-style env construction composes the isaacgym config itself,
    # so a GlobalHydra instance has to be live. initialize_config_dir, not
    # initialize: the latter rejects an absolute path.
    hydra.initialize_config_dir(
        config_dir=str(
            pathlib.Path(__file__).parent.joinpath("manifeel", "config").resolve()
        ),
        version_base=None,
    )
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(override)))
    workspace = hydra.utils.get_class(cfg._target_)(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(torch.device(device))
    policy.eval()
    return policy, cfg


def _print_header(checkpoint, out, cfg, options, task_spec):
    print("=" * 78)
    print(f"COLLECT MI DATASET  task={cfg.task.name}  spec={task_spec.name}")
    print(f"  checkpoint={checkpoint}")
    print(f"  out={out}")
    print(
        f"  episodes={options.num_episodes} in {options.num_batches} batches of "
        f"{options.num_envs}  start_seed={options.start_seed}"
    )
    print(
        f"  episode ends at first_success+{options.success_tail_steps} or "
        f"{options.max_steps} steps, whichever is first  "
        f"contact_threshold={options.contact_threshold}"
    )
    print(f"  channels={sorted(options.obs_channels)}")
    print("=" * 78)


def _sidecar(
    checkpoint,
    cfg,
    options,
    task_spec,
    control_dt,
    labels,
    provenance,
    merged,
    batches,
    elapsed,
    contact_stats,
):
    channels = list(merged["channels"])
    dead = [name for name, row in contact_stats.items() if row["live_fraction"] == 0.0]
    return {
        "contact_stats": contact_stats,
        # A channel that never fires carries exactly zero information. That is a
        # property of the task (no robot body reaches the socket in peg
        # insertion), not of the collector -- but an MI of 0.0 read off one of
        # these is not a measurement, and the reader has to be told which.
        "contact_channels_never_live": dead,
        # Reader requirement: without control_dt, timestamps fall back to step index.
        "control_dt": control_dt,
        # Analyzer requirement: all three pair labels must be present.
        "contact_pairs": labels,
        "contact_pair_bodies": provenance,
        "contact_force_convention": (
            "force_world is the force ON the bodies_a group FROM the bodies_b "
            "group, summed over both groups. contact_force_pairwise is exactly "
            "antisymmetric, so the direction is a labelling choice."
        ),
        "contact_threshold": options.contact_threshold,
        "contact_state_dim": CONTACT_STATE_DIM,
        "contact_state_layout": ["in_contact", "force_world_x", "force_world_y",
                                 "force_world_z"],
        "env": task_spec.env_name,
        "task_name": str(cfg.task.name),
        "task_spec": task_spec.name,
        "checkpoint": os.path.abspath(checkpoint),
        "policy_target": str(cfg.policy._target_),
        "shape_meta": OmegaConf.to_container(cfg.task.shape_meta, resolve=True),
        "n_obs_steps": int(cfg.n_obs_steps),
        "n_action_steps": int(cfg.n_action_steps),
        "use_ema": bool(cfg.training.use_ema),
        "num_episodes": merged["num_episodes"],
        "num_steps": merged["num_steps"],
        "num_success": merged["num_success"],
        "num_failure": merged["num_episodes"] - merged["num_success"],
        "max_steps": options.max_steps,
        "success_tail_steps": options.success_tail_steps,
        "num_envs": options.num_envs,
        "start_seed": options.start_seed,
        "state_full_layout": [list(x) for x in task_spec.state_full_layout()],
        "roles": writer_mod.channel_roles(channels),
        "analyzer": task_spec.analyzer_recommendations(),
        "batches": batches,
        "wall_clock_s": round(elapsed, 1),
    }


def _remove_if_empty(path):
    shards = os.path.join(path, "shards")
    if os.path.isdir(shards) and not os.listdir(shards):
        os.rmdir(shards)
    if os.path.isdir(path) and not os.listdir(path):
        os.rmdir(path)


if __name__ == "__main__":
    main()
