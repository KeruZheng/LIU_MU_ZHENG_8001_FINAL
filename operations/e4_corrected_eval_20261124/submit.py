"""Submit one isolated corrected evaluation to the existing SICC environment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from evaluate import atomic, digest

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--release', required=True)
parser.add_argument('--partition', default='gpu_batch')
args = parser.parse_args()
project = args.project.resolve()
root = project / 'releases' / args.release
previous = project / 'operations/e4_parallel_20261123'
graphics = json.loads((previous / 'nvidia-egl-runtime.json').read_text())
libraries = json.loads((previous / 'runtime-libraries.json').read_text())['library_paths']
env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', WANDB_MODE='disabled',
    OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
    LD_LIBRARY_PATH=':'.join([graphics['library_path']] + libraries),
    __EGL_VENDOR_LIBRARY_FILENAMES=graphics['vendor_file'])
with (root / 'submission.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = root / 'submission.json'
    if receipt.exists():
        print(receipt.read_text())
        raise SystemExit(0)
    for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
        assert digest(root / name) == expected, name
    name = 'robotic_e4_corrected_s0'
    active = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()), '--name', name, '-o', '%i'], text=True).strip()
    assert not active, 'Reconcile existing job before submitting: ' + active
    inputs = root / 'inputs/equidiff_s0'
    command = [str(project / 'envs/e4_20261123/bin/python'), '-u', str(root / 'evaluate.py'),
        'suite', '--upstream', str(root / 'upstream'), '--config', str(inputs / 'config.yaml'),
        '--weights', str(inputs / 'final_ema.pt'), '--metadata-dir', str(inputs / 'metadata'),
        '--output', str(root / 'results/equidiff_s0'), '--training-seed', '0']
    wrapped = 'nvidia-smi && exec ' + shlex.join(command)
    submit = ['sbatch', '--parsable', '--account=qingbiaoli', '--partition=' + args.partition,
        '--time=20:00:00', '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8',
        '--mem=40G', '--export=ALL', '--job-name', name, '--chdir', str(root),
        '--output', str(root / 'logs/slurm_%j.out'), '--wrap', wrapped]
    if args.partition == 'gpu_batch':
        submit[1:1] = ['--exclude=um-gpu03,um-gpu04']
    value = subprocess.run(submit, env=env, capture_output=True, text=True)
    if value.returncode:
        raise RuntimeError(value.stderr)
    job = value.stdout.strip().split(';')[0]
    assert job.isdigit(), value.stdout
    atomic(receipt, {'job_id': job, 'time': time.time(), 'command': submit,
                    'release': args.release, 'scope': 'Corrected D1/D2/D0 final EMA evaluation'})
    print(receipt.read_text())
