"""Run one fresh, full official Push-T reference with verified offloading."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'GPU execution requires Slurm'
    root = args.project / 'releases/official_dp_s42_20261120_v1'
    upstream = args.project / 'upstream/diffusion_policy'
    sys.path.insert(0, str(upstream))
    sys.path.insert(0, str(args.project / 'operations/parallel_20261119'))
    from transfer_guard import atomic, now
    from convergence import verify
    verify(root)
    for relative, expected in json.loads((root / 'upstream_files.json').read_text()).items():
        assert hashlib.sha256((upstream / relative).read_bytes()).hexdigest() == expected, relative
    import gym
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
    assert gym.__version__ == '0.21.0', gym.__version__
    assert torch.cuda.is_available()
    subprocess.run(['nvidia-smi'], check=True)
    x = torch.ones((32, 32), device='cuda')
    assert (x @ x).sum().item() == 32 ** 3
    torch.cuda.synchronize()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = OmegaConf.load(root / 'config.yaml')
    assert cfg.training.num_epochs == 3050 and not cfg.training.debug
    assert cfg.training.max_train_steps is None and cfg.training.max_val_steps is None
    assert cfg.training.seed == 42 and cfg.task.dataset.max_train_episodes == 90
    run = 'official_dp_s42'
    output = root / 'runs' / run
    # Refuse accidental duplicate or incorrect upstream resume semantics.
    output.mkdir(exist_ok=False)
    cfg.task.dataset.zarr_path = str(args.project / 'data/pusht/pusht_cchi_v7_replay.zarr')
    cfg.training.resume = False
    cfg.logging.mode = 'disabled'
    cfg.logging.name = run
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

    info = {'state': 'training', 'time': now(), 'job_id': os.environ['SLURM_JOB_ID'],
            'run': run, 'target_epochs': 3050, 'seed': 42,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gym': gym.__version__,
            'official_config_sha256': hashlib.sha256((root / 'config.yaml').read_bytes()).hexdigest(),
            'scope': 'Official published 90-demo Push-T image recipe, one seed; separate from primary matrix'}
    begin = time.monotonic()
    with (output / 'compute.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(root / 'results/lane_status.json', info)
        atomic(output / 'started.json', info)
        try:
            workspace = AtomicWorkspace(cfg, output_dir=str(output))
            workspace.run()
            assert workspace.epoch == 3050, 'Incomplete reference training'
            # Preserve the terminal state as well as the upstream top-5 and latest.
            workspace.save_checkpoint(tag='final')
            info.update(state='complete', time=now(), seconds=time.monotonic() - begin,
                        epoch=workspace.epoch, global_step=workspace.global_step,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated())
            atomic(output / 'completed.json', info)
            atomic(root / 'results/lane_status.json', info)
            checkpoints = sorted(str(p.relative_to(output)) for p in output.glob('checkpoints/*.ckpt'))
            atomic(output / 'archive_ready.json', {'state': 'complete', 'study': 'official_reference',
                                                   'checkpoints': checkpoints})
        except Exception as exc:
            info.update(state='failed', time=now(), error=repr(exc))
            atomic(root / 'results/lane_status.json', info)
            raise
    atomic(args.project / 'operations/parallel_20261119/requests' / (root.name + '__' + run + '.json'),
           {'release': root.name, 'run': run})
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
