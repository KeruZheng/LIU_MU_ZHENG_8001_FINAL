"""One seed, two unchanged E4 methods, with preflight and verified archive gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from prepare_data import atomic, digest
from transfer_guard import gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    root, project = args.root.resolve(), args.project.resolve()
    lane = json.loads((root / 'lane.json').read_text())
    seed = lane['seed']
    status = {'state': 'starting', 'seed': seed, 'completed_runs': 0, 'planned_runs': 2,
              'supervisor_pid': os.getpid(), 'job_id': os.environ.get('SLURM_JOB_ID')}

    def update(**kwargs):
        status.update(kwargs, heartbeat=time.time())
        atomic(root / 'results/lane_status.json', status)

    def invoke(method, mode):
        for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
            assert digest(root / name) == expected, name
        name = f'{method}_square_d0_s{seed}'
        update(state='preflight' if mode == 'preflight' else 'training_and_transfer', run=name)
        env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4',
                   OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled', PYTHONPATH=str(root / 'upstream'))
        with (root / 'logs' / f'{mode}_{name}.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(root / 'run.py'), '--root', str(root),
                '--codes', str(project), '--method', method, '--seed', str(seed), '--mode', mode],
                cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            while child.poll() is None:
                update()
                time.sleep(10)
            assert child.returncode == 0, f'{name} {mode} exit {child.returncode}'
            update(child_pid=None)

    with (root / 'results/compute.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(root / 'results/environment.json', {'job_id': os.environ.get('SLURM_JOB_ID'),
            'node': os.environ.get('SLURMD_NODENAME'), 'python': sys.executable,
            'nvidia_smi': subprocess.check_output(['nvidia-smi'], text=True),
            'freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)})
        try:
            for method in ('equidiff', 'dp'):
                invoke(method, 'preflight')
            left, right = [json.loads((root / f'preflight/{m}_square_d0_s{seed}/result.json').read_text())
                           for m in ('equidiff', 'dp')]
            assert left['state'] == right['state'] == 'complete'
            for key in ('target_updates', 'batch_size', 'train_episode_ids', 'validation_episode_ids', 'training_input_sha256'):
                assert left[key] == right[key] == lane['reference_runtime'][key], key
            atomic(root / 'results/preflight_passed.json', {'state': 'complete', 'seed': seed,
                'time': time.time(), 'matched_local_protocol': True, 'target_updates': left['target_updates']})
            for method in ('equidiff', 'dp'):
                invoke(method, 'train')
                name = f'{method}_square_d0_s{seed}'
                output = root / 'runs' / name
                assert json.loads((output / 'completed.json').read_text())['state'] == 'complete'
                for domain in ('d0', 'd1', 'd2'):
                    assert json.loads((output / f'transfer_{domain}/scores.json').read_text())['episodes'] == 100
                checkpoints = sorted(str(p.relative_to(output)) for p in (output / 'checkpoints').glob('*.ckpt'))
                assert checkpoints
                atomic(output / 'archive_ready.json', {'state': 'complete', 'checkpoints': checkpoints})
                update(state='waiting_verified_archive', run=name)
                # The existing archive worker copies and hashes every file before removing models.
                gate(project, root.name, name)
                update(completed_runs=status['completed_runs'] + 1)
            update(state='complete', run=None)
        except BaseException as exc:
            update(state='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
