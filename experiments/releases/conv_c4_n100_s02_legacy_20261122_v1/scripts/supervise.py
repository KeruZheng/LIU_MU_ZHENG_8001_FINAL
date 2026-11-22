"""Local C4 follow-up curves with singleton ownership and retained checkpoints."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def atomic(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = json.loads((root / 'manifests/lane.json').read_text())
    frozen = json.loads((root / 'manifests/frozen_files.json').read_text())
    env = dict(os.environ, PYTHONPATH=str(root / 'src'), SDL_VIDEODRIVER='dummy',
               PYGAME_HIDE_SUPPORT_PROMPT='1', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               ROBOTIC_GPU_MEMORY_FRACTION='0.70')
    status = {'supervisor_pid': os.getpid(), 'state': 'preflight'}

    def update(**kwargs):
        status.update(kwargs, heartbeat=time.time())
        atomic(root / 'results/supervisor_status.json', status)

    def invoke(command, name):
        for relative, expected in frozen.items():
            assert digest(root / relative) == expected, relative
        with (root / 'logs' / (name + '.log')).open('a') as log:
            child = subprocess.Popen([sys.executable] + command, cwd=root, env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(name=name, child_pid=child.pid)
            while child.poll() is None:
                update()
                time.sleep(10)
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}')

    with (root / 'results/supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        update()
        try:
            import torch
            assert torch.cuda.is_available()
            x = torch.ones((32, 32), device='cuda')
            assert (x @ x).sum().item() == 32 ** 3
            update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda)
            del x
            torch.cuda.empty_cache()
            invoke(['scripts/validate_c4.py', '--output', 'results/native_validation.json'], 'native_validation')
            for run in plan['runs']:
                directory = root / 'runs' / run
                directory.mkdir(exist_ok=True)
                update(run=run)
                for step in plan['milestones']:
                    p = directory / 'status.json'
                    old = json.loads(p.read_text()) if p.exists() else {}
                    if old.get('state') != 'complete' or old.get('step', 0) < step:
                        update(state='training', target_step=step)
                        cmd = ['-m', 'pusht_study.train', '--config', f'configs/{run}_{step}.json', '--output', str(directory)]
                        if (directory / 'last.pt').exists():
                            cmd.append('--resume')
                        invoke(cmd, run + '_train')
                    report = directory / f'validation_step_{step}.json'
                    old = json.loads(report.read_text()) if report.exists() else {}
                    if old.get('state') != 'complete':
                        update(state='evaluating', target_step=step)
                        cmd = ['-m', 'pusht_study.evaluate_c4', '--checkpoint', str(directory / f'ema_step_{step}.pt'),
                               '--output', str(report), '--validation-report', 'results/native_validation.json',
                               '--split', 'validation', '--batch-size', '8']
                        if report.exists():
                            cmd.append('--resume')
                        invoke(cmd, run + f'_validation_{step}')
                    result = json.loads(report.read_text())
                    assert result['state'] == 'complete' and result['summary']['episodes'] == 100
                update(state='verifying_local', name=run + '_local_inventory')
                models = [directory / 'last.pt'] + [directory / f'ema_step_{s}.pt' for s in plan['milestones']]
                records = []
                for p in models:
                    records.append({'name': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)})
                    with p.open('rb') as stream:
                        os.fsync(stream.fileno())
                atomic(directory / 'local_inventory.json', {'state': 'retained_locally', 'files': records,
                    'scope': 'Original local artifacts; no second backup copy claimed; no deletion'})
            update(state='complete', name='', run='', child_pid=None)
        except Exception as exc:
            update(state='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
