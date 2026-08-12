"""
Usage:
python eval.py \
--checkpoint data/outputs/vision_pih_50/44/checkpoints/latest_epoch200.ckpt \
-o data/outputs/vision_pih_50/44/eval_epoch200 \
-d cuda:0 \
-x task.env_runner.test_start_seed=200000
"""

import isaacgym

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import numpy as np
import dill
import wandb
import json
from omegaconf import OmegaConf
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# the checkpoint's cfg may still carry unresolved ${eval:'...'} interpolations
OmegaConf.register_new_resolver("eval", eval, replace=True)

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-n', '--cfg_name', default=None,
              help='Unused; kept for backwards compatibility. The config comes from the checkpoint.')
@click.option('-d', '--device', default='cuda:0')
@click.option('-x', '--override', multiple=True,
              help='dotlist override applied to the checkpoint cfg, e.g. task.env_runner.test_start_seed=200000')
@click.option('--yes', is_flag=True, help='Overwrite an existing output dir without prompting.')
def main(checkpoint, output_dir, cfg_name, device, override, yes):
    if os.path.exists(output_dir) and not yes:
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ManifeelRunner.__init__ composes the isaacgym config itself, so a GlobalHydra
    # instance has to be live even though the workspace cfg comes from the checkpoint.
    # initialize_config_dir, not initialize: the latter rejects an absolute path.
    hydra.initialize_config_dir(config_dir=str(pathlib.Path(__file__).parent.joinpath(
        'manifeel', 'config').resolve()), version_base=None)

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    # Take the training cfg rather than recomposing it: this is what guarantees the
    # policy shape_meta and the eval env match the run that produced the weights.
    cfg = payload['cfg']
    if override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(override)))
    print(OmegaConf.to_yaml(cfg.task.env_runner))
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            if isinstance(value, np.integer):
                value = int(value)
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
