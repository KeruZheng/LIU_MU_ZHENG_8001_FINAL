"""Validated image RQ3 lane with per-run archive gating."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root.parents[1]
    sys.path.insert(0, str(project / 'operations/parallel_20261119'))
    from convergence import verify
    from transfer_guard import atomic, gate, now
    assert os.environ.get('SLURM_JOB_ID')
    env = dict(os.environ, PYTHONPATH=str(root / 'src'), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               SDL_VIDEODRIVER='dummy', PYGAME_HIDE_SUPPORT_PROMPT='1', ROBOTIC_GPU_MEMORY_FRACTION='0.85')
    plan = json.loads((root / 'manifests/lane.json').read_text())

    def invoke(command, log):
        verify(root)
        with (root / 'logs' / log).open('a') as stream:
            subprocess.run([sys.executable] + command, cwd=root, env=env, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)

    with (root / 'results/lane.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(root)
        status = {'state': 'preflight', 'time': now(), 'job_id': os.environ['SLURM_JOB_ID']}
        atomic(root / 'results/lane_status.json', status)
        try:
            invoke(['-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_image_rq3.py', '-v'], 'image_tests.log')
            invoke(['scripts/validate_c4.py', '--output', 'results/native_validation.json'], 'native_validation.log')
            invoke(['scripts/image_preflight.py'], 'image_preflight.log')
            assert json.loads((root / 'results/image_preflight.json').read_text())['passed']
            for run in plan['runs']:
                receipt = project / 'operations/parallel_20261119/receipts' / (root.name + '__' + run + '.json')
                if receipt.exists():
                    continue
                directory = root / 'runs' / run
                directory.mkdir(exist_ok=True)
                with (directory / 'compute.lock').open('a') as compute:
                    fcntl.flock(compute, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    status.update(state='training', run=run, target_step=20000, time=now())
                    atomic(root / 'results/lane_status.json', status)
                    p = directory / 'status.json'
                    previous = json.loads(p.read_text()) if p.exists() else {}
                    if previous.get('state') != 'complete' or previous.get('step', 0) < 20000:
                        cmd = ['-m', 'pusht_study.train', '--config', 'configs/' + run + '.json', '--output', str(directory)]
                        if (directory / 'last.pt').exists():
                            cmd.append('--resume')
                        invoke(cmd, run + '_train.log')
                    for step in plan['milestones']:
                        report = directory / f'test_step_{step}.json'
                        if report.exists() and json.loads(report.read_text()).get('state') == 'complete':
                            continue
                        status.update(state='evaluating', target_step=step, time=now())
                        atomic(root / 'results/lane_status.json', status)
                        cmd = ['-m', 'pusht_study.evaluate_c4', '--checkpoint', str(directory / f'ema_step_{step}.pt'),
                               '--output', str(report), '--validation-report', 'results/native_validation.json',
                               '--split', 'test', '--batch-size', '8']
                        if report.exists():
                            cmd.append('--resume')
                        invoke(cmd, run + '_test.log')
                    for step in plan['milestones']:
                        result = json.loads((directory / f'test_step_{step}.json').read_text())
                        assert result['state'] == 'complete' and result['summary']['episodes'] == 100
                    atomic(directory / 'archive_ready.json', {'state': 'complete', 'study': 'image_RQ3_visible_goal',
                        'checkpoints': ['last.pt'] + [f'ema_step_{s}.pt' for s in plan['milestones']]})
                status.update(state='waiting_archive', time=now())
                atomic(root / 'results/lane_status.json', status)
                gate(project, root.name, run)
            atomic(root / 'results/lane_status.json', dict(status, state='complete', time=now(), runs=len(plan['runs'])))
        except Exception as exc:
            atomic(root / 'results/lane_status.json', dict(status, state='failed', time=now(), error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
