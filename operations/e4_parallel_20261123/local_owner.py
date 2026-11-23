"""Adopt the live seed-0 trainer, then dispatch only DP seed 0."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def identity(pid):
    base = Path('/proc') / str(pid)
    try:
        stat = (base / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'start': stat[19], 'state': stat[0], 'ppid': int(stat[1]),
                'command': (base / 'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return None


def same_live(pid, original):
    current = identity(pid)
    return current and current['start'] == original['start'] and current['state'] != 'Z'


def validate_handoff(root, old_pid, child_pid):
    state = json.loads((root / 'results/supervisor_status.json').read_text())
    run = root / 'runs/equidiff_square_d0_s0'
    training = json.loads((run / 'status.json').read_text())
    old, child = identity(old_pid), identity(child_pid)
    assert old and child and child['ppid'] == old_pid
    assert str(root / 'supervise.py') in old['command']
    assert str(root / 'run.py') in child['command'] and '--seed 0' in child['command']
    assert state['supervisor_pid'] == old_pid and state['child_pid'] == child_pid
    assert state['run'] == 'equidiff_square_d0_s0' and training['epoch'] < 450
    assert not (root / 'runs/dp_square_d0_s0').exists()
    assert not any(root.joinpath('runs').glob('*_s[12]'))
    for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    return state, old, child


def completed(root, name):
    output = root / 'runs' / name
    value = json.loads((output / 'completed.json').read_text())
    assert value['state'] == 'complete' and value['epoch'] == 500
    assert value['global_step'] == value['target_updates'] == 112000
    for domain in ('d0', 'd1', 'd2'):
        assert json.loads((output / f'transfer_{domain}/scores.json').read_text())['episodes'] == 100
    assert (output / 'local_inventory.json').exists()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--codes', type=Path, required=True)
    parser.add_argument('--old-pid', type=int, required=True)
    parser.add_argument('--child-pid', type=int, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    state, old, child = validate_handoff(root, args.old_pid, args.child_pid)
    if args.check_only:
        print('Validated supervisor-only handoff; trainer remains live')
        return
    assert not args.receipt.exists()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.receipt, {'state': 'prepared', 'old_pid': args.old_pid, 'child_pid': args.child_pid,
                         'old_identity': old, 'child_identity': child, 'time': time.time()})
    # Signal the scheduler PID only. Its training child owns separate log files
    # and survives; never signal the process group or the training PID.
    os.kill(args.old_pid, signal.SIGTERM)
    with (root / 'results/compute.lock').open('a') as lock:
        deadline = time.monotonic() + 20
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Previous owner did not release its queue lock')
                time.sleep(0.2)
        assert same_live(args.child_pid, child), 'Adopted training process missing'
        state.update(supervisor_pid=os.getpid(), planned_runs=2, completed_runs=0,
                     seed_ownership=[0], remote_seeds=[1, 2], adopted_child_pid=args.child_pid)

        def update(**kwargs):
            state.update(kwargs, heartbeat=time.time())
            atomic(root / 'results/supervisor_status.json', state)

        (root / 'results/supervisor.pid').write_text(str(os.getpid()) + '\n')
        atomic(args.receipt, {'state': 'adopted', 'time': time.time(), 'old_pid': args.old_pid,
            'new_pid': os.getpid(), 'child_pid': args.child_pid, 'child_identity': identity(args.child_pid),
            'frozen_source_unchanged': True, 'local_seeds': [0], 'remote_seeds': [1, 2]})
        try:
            update()
            while same_live(args.child_pid, child):
                update()
                time.sleep(10)
            completed(root, 'equidiff_square_d0_s0')
            update(completed_runs=1, child_pid=None, method='dp', run='dp_square_d0_s0')
            env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', WANDB_MODE='disabled', PYTHONPATH=str(root / 'upstream'))
            with (root / 'logs/dp_square_d0_s0.log').open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', str(root / 'run.py'), '--root', str(root),
                    '--codes', str(args.codes.resolve()), '--method', 'dp', '--seed', '0', '--mode', 'train'],
                    cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                update(child_pid=process.pid)
                while process.poll() is None:
                    update()
                    time.sleep(10)
                assert process.returncode == 0, f'DP seed 0 exit {process.returncode}'
            completed(root, 'dp_square_d0_s0')
            update(state='complete', child_pid=None, completed_runs=2)
        except BaseException as exc:
            update(state='failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
