"""Official DP engineering preflight; its scores are never scientific results."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Use Slurm'
    root = a.project / 'releases/official_dp_preflight_20261119_v1'
    upstream = a.project / 'upstream/diffusion_policy'
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(a.project / 'operations/parallel_20261119'))
    from transfer_guard import atomic
    from convergence import verify
    verify(root)
    for relative, expected in json.loads((root / 'upstream_files.json').read_text()).items():
        assert hashlib.sha256((upstream / relative).read_bytes()).hexdigest() == expected, relative
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
    assert torch.cuda.is_available()
    x = torch.ones((32, 32), device='cuda')
    assert (x @ x).sum().item() == 32 ** 3
    torch.cuda.synchronize()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = 'preflight_' + os.environ['SLURM_JOB_ID']
    output = root / 'runs' / run
    output.mkdir(exist_ok=False)
    cfg = OmegaConf.load(root / 'config.yaml')
    cfg.task.dataset.zarr_path = str(a.project / 'data/pusht/pusht_cchi_v7_replay.zarr')
    cfg.training.debug = True
    cfg.training.resume = False
    cfg.logging.mode = 'disabled'
    cfg.logging.name = run
    cfg.dataloader.num_workers = 2
    cfg.val_dataloader.num_workers = 2
    cfg.task.env_runner.n_envs = 2
    cfg.task.env_runner.n_train = 1
    cfg.task.env_runner.n_test = 1
    cfg.task.env_runner.n_train_vis = 1
    cfg.task.env_runner.n_test_vis = 1
    cfg.task.env_runner.test_start_seed = 900000
    cfg.checkpoint.topk.k = 1
    OmegaConf.save(cfg, output / 'effective_config.yaml')

    class AtomicWorkspace(TrainDiffusionUnetHybridWorkspace):
        def save_checkpoint(self, path=None, tag='latest', **kwargs):
            destination = Path(path) if path else self.get_checkpoint_path(tag)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + '.tmp')
            kwargs.pop('use_thread', None)
            super().save_checkpoint(path=temporary, use_thread=False, **kwargs)
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
            return str(destination)

    begin = time.monotonic()
    with (output / 'compute.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = {'job_id': os.environ['SLURM_JOB_ID'], 'gpu': torch.cuda.get_device_name(),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'official_config_sha256': hashlib.sha256((root / 'config.yaml').read_bytes()).hexdigest(),
                'scope': 'Two debug epochs, three training batches each; two rollout seeds; engineering only'}
        atomic(output / 'preflight.json', info)
        workspace = AtomicWorkspace(cfg, output_dir=str(output))
        workspace.run()
        assert workspace.epoch == 2, 'Incomplete debug workspace'
        info.update(state='complete', seconds=time.monotonic() - begin,
                    epoch=workspace.epoch, global_step=workspace.global_step,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())
        atomic(output / 'preflight_result.json', info)
        atomic(root / 'results/preflight.json', info)
        checkpoints = sorted(str(x.relative_to(output)) for x in output.glob('checkpoints/*.ckpt'))
        atomic(output / 'archive_ready.json', {'state': 'complete', 'study': 'engineering_only', 'checkpoints': checkpoints})
    atomic(a.project / 'operations/parallel_20261119/requests' / (root.name + '__' + run + '.json'),
           {'release': root.name, 'run': run})
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
