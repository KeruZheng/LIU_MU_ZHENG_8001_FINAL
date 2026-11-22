"""Own the complete E4 preparation, preflight and six-run local queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare_data import atomic, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--codes', type=Path, required=True)
    args = parser.parse_args()
    root, codes = args.root.resolve(), args.codes.resolve()
    status = {'supervisor_pid': os.getpid(), 'state': 'starting', 'completed_runs': 0, 'planned_runs': 6}
    def update(**kwargs):
        status.update(kwargs, heartbeat=time.time())
        atomic(root / 'results/supervisor_status.json', status)
    env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled', PYTHONPATH=str(root / 'upstream'))
    def invoke(command, logname):
        for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
            assert digest(root / name) == expected, name
        with (root / 'logs' / logname).open('x') as log:
            child = subprocess.Popen([sys.executable, '-u'] + command, cwd=root, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            while child.poll() is None:
                update()
                time.sleep(10)
            assert child.returncode == 0, f'{logname} exited {child.returncode}'
            update(child_pid=None)
    with (root / 'results/compute.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (root / 'results/supervisor.pid').write_text(str(os.getpid()) + '\n')
        try:
            sources = json.loads((root / 'data_sources.json').read_text())
            begin = time.monotonic()
            while True:
                ready, progress = True, {}
                for item in sources['files']:
                    name = Path(item['path']).stem
                    raw = codes / f'data/robomimic/datasets/{name}/{name}.hdf5'
                    p = raw if raw.exists() else raw.with_suffix('.hdf5.part')
                    size = p.stat().st_size if p.exists() else 0
                    progress[name] = {'bytes': size, 'expected': item['size']}
                    ready &= size == item['size']
                update(state='waiting_downloads', downloads=progress)
                if ready:
                    break
                if time.monotonic() - begin > 3600:
                    raise TimeoutError('Downloads incomplete after one hour; do not start training')
                time.sleep(10)
            update(state='verifying_and_converting_data')
            invoke([str(root / 'prepare_data.py'), '--codes', str(codes), '--sources', str(root / 'data_sources.json'),
                    '--output', str(root / 'results/data_ready.json')], 'prepare_data.log')
            for method in ('equidiff', 'dp'):
                update(state='preflight', method=method, seed=0)
                invoke([str(root / 'run.py'), '--root', str(root), '--codes', str(codes),
                        '--method', method, '--seed', '0', '--mode', 'preflight'], f'preflight_{method}.log')
                assert json.loads((root / 'preflight' / f'{method}_square_d0_s0/result.json').read_text())['state'] == 'complete'
            left = json.loads((root / 'preflight/equidiff_square_d0_s0/runtime.json').read_text())
            right = json.loads((root / 'preflight/dp_square_d0_s0/runtime.json').read_text())
            for key in ('target_updates', 'batch_size', 'train_episode_ids', 'validation_episode_ids', 'training_input_sha256'):
                assert left[key] == right[key], 'Unmatched comparison: ' + key
            atomic(root / 'results/preflight_passed.json', {'passed': True, 'time': time.time(),
                'target_updates': left['target_updates'], 'batch_size': left['batch_size'],
                'scope': 'Both models and all three domains passed; no scientific claim from short rollouts'})
            for seed in (0, 1, 2):
                for method in ('equidiff', 'dp'):
                    run = f'{method}_square_d0_s{seed}'
                    update(state='training_and_transfer', method=method, seed=seed, run=run)
                    invoke([str(root / 'run.py'), '--root', str(root), '--codes', str(codes),
                            '--method', method, '--seed', str(seed), '--mode', 'train'], run + '.log')
                    output = root / 'runs' / run
                    assert json.loads((output / 'completed.json').read_text())['state'] == 'complete'
                    for domain in ('d0', 'd1', 'd2'):
                        assert json.loads((output / ('transfer_' + domain) / 'scores.json').read_text())['episodes'] == 100
                    update(completed_runs=status['completed_runs'] + 1)
            update(state='complete', child_pid=None)
        except BaseException as exc:
            update(state='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
