"""Continue a terminated seed lane in a new release, never its active directory."""
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
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--project', type=Path, required=True)
    a = p.parse_args()
    root, project = a.root.resolve(), a.project.resolve()
    plan = json.loads((root / 'continuation.json').read_text())
    seed = plan['seed']
    source = project / 'releases' / plan['source_release']
    status = {'state': 'starting', 'seed': seed, 'completed_runs': 0,
              'job_id': os.environ.get('SLURM_JOB_ID'), 'source_job': plan['source_job']}

    def update(**values):
        status.update(values, heartbeat=time.time())
        atomic(root / 'results/lane_status.json', status)

    def invoke(command, logname):
        with (root / 'logs' / logname).open('x') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            update(child_pid=child.pid)
            while child.poll() is None:
                update()
                time.sleep(10)
            assert child.returncode == 0, f'{logname}: exit {child.returncode}'
            update(child_pid=None)

    with (root / 'results/compute.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            active = subprocess.check_output(['squeue', '-h', '-j', plan['source_job'], '-o', '%T'], text=True).strip()
            assert not active, 'Source job must terminate before checkpoint snapshot'
            for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
                assert digest(root / name) == expected, name
            for method in ('equidiff', 'dp'):
                name = f'{method}_square_d0_s{seed}'
                old_run = source / 'runs' / name
                if (old_run / 'completed.json').exists():
                    raise RuntimeError('Source completed unexpectedly; reconcile before duplicate evaluation: '+name)
                checkpoint = old_run / 'checkpoints/latest.ckpt'
                command = [sys.executable, '-u', str(root / 'run.py'), '--root', str(root),
                           '--codes', str(project), '--method', method, '--seed', str(seed), '--mode', 'train']
                if checkpoint.exists():
                    snapshot = root / 'resume_inputs' / f'{method}_s{seed}.ckpt'
                    snapshot.parent.mkdir(exist_ok=True)
                    # Old atomic checkpoint inode is immutable after the source job exits.
                    os.link(checkpoint, snapshot)
                    assert digest(snapshot) == digest(checkpoint)
                    command += ['--resume-checkpoint', str(snapshot)]
                    update(state='resuming', run=name, checkpoint=str(snapshot))
                else:
                    assert method == 'dp' and not old_run.exists(), 'Missing required recovery checkpoint'
                    update(state='training', run=name)
                invoke(command, f'train_{name}.log')
                output = root / 'runs' / name
                completed = json.loads((output / 'completed.json').read_text())
                assert completed['epoch'] == 500 and completed['global_step'] == 112000
                update(state='exporting_for_corrected_evaluation', run=name)
                inputs = output / 'inference'
                invoke([sys.executable, str(root / 'export.py'), '--release', str(root),
                        '--method', method, '--seed', str(seed), '--output', str(inputs)], f'export_{name}.log')
                update(state='corrected_evaluation', run=name)
                invoke([sys.executable, '-u', str(root / 'evaluate.py'), 'suite',
                        '--upstream', str(root / 'upstream'), '--config', str(inputs / 'config.yaml'),
                        '--weights', str(inputs / 'final_ema.pt'), '--metadata-dir', str(inputs / 'metadata'),
                        '--output', str(output / 'corrected_eval'), '--training-seed', str(seed)], f'eval_{name}.log')
                assert json.loads((output / 'corrected_eval/completed.json').read_text())['state'] == 'complete'
                checkpoints = [str(p.relative_to(output)) for p in (output / 'checkpoints').glob('*.ckpt')]
                checkpoints.append('inference/final_ema.pt')
                atomic(output / 'archive_ready.json', {'state': 'complete', 'checkpoints': checkpoints})
                update(state='waiting_verified_archive')
                gate(project, root.name, name)
                update(completed_runs=status['completed_runs'] + 1)
            update(state='complete', run=None)
        except BaseException as exc:
            update(state='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
