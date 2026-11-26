"""Validate actual saved tensors, optimizer steps, EMA and LR continuation."""
import argparse
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--method', choices=['equidiff', 'dp'], default='equidiff')
p.add_argument('--seed', type=int, default=0)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(a.source.resolve()))
sys.path.insert(0, str(a.source.resolve() / 'upstream'))
import hydra
import torch
from omegaconf import OmegaConf
from resume_support import restore
from prepare_data import atomic

torch.set_num_threads(4)
cfg = OmegaConf.load(a.source / f'configs/{a.method}_s{a.seed}.yaml')
w = hydra.utils.get_class(cfg._target_)(cfg, output_dir=str(a.output))
r = restore(w, a.checkpoint, 224, a.output / 'resume_receipt.json')
ema = hydra.utils.instantiate(cfg.ema, model=w.ema_model)
assert ema.optimization_step == w.global_step
assert ema.get_decay(ema.optimization_step) > 0.99
from equi_diffpo.model.common.lr_scheduler import get_scheduler
lr_before = [g['lr'] for g in w.optimizer.param_groups]
scheduler = get_scheduler(cfg.training.lr_scheduler, w.optimizer,
    num_warmup_steps=cfg.training.lr_warmup_steps, num_training_steps=112000,
    last_epoch=w.global_step-1)
assert scheduler.last_epoch == w.global_step
assert all(abs(x-y) < 1e-12 for x,y in zip(lr_before,scheduler.get_last_lr()))
r.update(state='passed', lr_before=lr_before, lr_after=scheduler.get_last_lr(),
         ema_decay=ema.get_decay(ema.optimization_step))
atomic(a.output / 'passed.json', r)
print(json.dumps(r, indent=2))
