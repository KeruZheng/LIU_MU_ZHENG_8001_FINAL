"""Measure the unchanged C8 recipe, then use the allocation for a new C4 curve."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    parser.add_argument('--lane')
    args = parser.parse_args()
    ops = args.project / 'operations/parallel_20261119'
    sys.path.insert(0, str(ops))
    from convergence import verify
    from transfer_guard import atomic
    import torch
    assert os.environ.get('SLURM_JOB_ID'), 'GPU work requires Slurm'
    assert torch.cuda.is_available(), 'CUDA preflight failed'
    x = torch.ones((32, 32), device='cuda')
    assert (x @ x).sum().item() == 32 ** 3
    torch.cuda.synchronize()
    props = torch.cuda.get_device_properties(0)
    root = args.project / 'releases/c8_speed_20261119_v1'
    verify(root)
    run = 'calibration_' + os.environ['SLURM_JOB_ID']
    output = root / 'runs' / run
    output.mkdir(exist_ok=False)
    info = {'gpu': props.name, 'total_memory': props.total_memory,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'job_id': os.environ['SLURM_JOB_ID'], 'study': 'runtime_calibration_only'}
    atomic(output / 'preflight.json', info)
    if props.total_memory >= 24 * 1024 ** 3:
        env = dict(os.environ, PYTHONPATH=str(root / 'src'), OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', SDL_VIDEODRIVER='dummy',
                   PYGAME_HIDE_SUPPORT_PROMPT='1', ROBOTIC_GPU_MEMORY_FRACTION='0.85')
        command = [sys.executable, '-m', 'pusht_study.train', '--config',
                   'configs/calibration.json', '--output', str(output), '--device', 'cuda']
        with (root / 'logs' / (run + '.log')).open('a') as log:
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        steady = [r for r in rows if r['step'] > 20]
        info.update(state='complete', median_update_seconds=statistics.median(r['update_seconds'] for r in steady),
                    elapsed_seconds=rows[-1]['elapsed_seconds'], steps=rows[-1]['step'])
        atomic(output / 'calibration.json', info)
        atomic(output / 'archive_ready.json', {'state': 'complete',
               'checkpoints': ['last.pt', 'ema_step_200.pt'], 'study': 'runtime_calibration_only'})
        atomic(ops / 'requests' / (root.name + '__' + run + '.json'), {'release': root.name, 'run': run})
    else:
        atomic(output / 'calibration.json', dict(info, state='skipped_insufficient_memory',
               reason='Keep full C8 batch64 recipe unchanged; proceed with C4 lane if supplied'))
    if args.lane:
        os.execv(sys.executable, [sys.executable, str(ops / 'convergence.py'), '--root',
                                 str(args.project / 'releases' / args.lane)])


if __name__ == '__main__':
    main()
