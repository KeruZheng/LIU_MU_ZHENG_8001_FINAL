"""Full-size GPU backward and native rollout checks before image RQ3."""
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import torch
from pusht_study import ROOT
from pusht_study.data import PushTSequences
from pusht_study.models import make_model
from pusht_study.evaluate_c4 import rollout_batch
from pusht_study.train import atomic_json, make_scheduler


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Run GPU checks inside Slurm'
    assert torch.cuda.is_available()
    subprocess.run(['nvidia-smi'], check=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = json.loads((ROOT / 'manifests/lane.json').read_text())
    config = json.loads((ROOT / 'configs' / (plan['runs'][0] + '.json')).read_text())
    split = json.loads((ROOT / config['split']).read_text())
    data = PushTSequences(ROOT / config['data'], split, config['demos'], config['horizon'],
                          config['obs_steps'], config['action_steps'], config['rotated'], cache=True)
    results = []
    for method in ('dp', 'equidiff_c4', 'equidiff_c4_gate'):
        torch.manual_seed(900000)
        cfg = dict(config, method=method)
        model = make_model(cfg).cuda()
        ema = copy.deepcopy(model).requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], betas=(0.95, 0.999), weight_decay=1e-6)
        scheduler = make_scheduler(cfg)
        torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            batch = {k: v.cuda() for k, v in data.get_batch(torch.arange(64) + step * 64).items()}
            noise = torch.randn_like(batch['action'])
            t = torch.randint(cfg['diffusion_steps'], (64,), device='cuda')
            noisy = scheduler.add_noise(batch['action'], noise, t)
            loss = torch.nn.functional.mse_loss(model(noisy, t, batch['image'], batch['agent_pos']), noise)
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        ema.load_state_dict(model.state_dict())
        ema.eval()
        rows = rollout_batch(ema, cfg, [(900000, 0), (900001, 1)], 'cuda', max_steps=8)
        assert len(rows) == 2 and all(0 <= r['max_coverage'] <= 1.000001 for r in rows)
        results.append({'method': method, 'loss': loss.item(), 'rollout': rows,
                        'peak_gpu_bytes': torch.cuda.max_memory_allocated()})
        del model, ema, optimizer, scheduler, batch, noise, t, noisy, loss
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(ROOT / 'results/image_preflight.json', {'passed': True, 'job_id': os.environ['SLURM_JOB_ID'],
        'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'scope': 'engineering only; full-size batch64 backward and two short validation rollouts', 'models': results})


if __name__ == '__main__':
    main()
