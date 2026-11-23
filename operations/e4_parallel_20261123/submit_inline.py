"""Run the corrected EGL check and full E4 lane in the same GPU allocation."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True, type=Path)
parser.add_argument('--account', required=True)
args = parser.parse_args()
project = args.project.resolve()
work = project / 'operations/e4_parallel_20261123'
graphics = json.loads((work / 'nvidia-egl-runtime.json').read_text())
libraries = json.loads((work / 'runtime-libraries.json').read_text())['library_paths']
env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', OPENBLAS_NUM_THREADS='1',
    OMP_NUM_THREADS='4', LD_LIBRARY_PATH=':'.join([graphics['library_path']] + libraries),
    __EGL_VENDOR_LIBRARY_FILENAMES=graphics['vendor_file'])
diagnostic = work / 'egl_diagnose_v3.py'
with (work / 'inline_submission.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed in (1, 2):
        root = project / f'releases/e4_square_d0_s{seed}_sicc_20261123_v2'
        receipt = work / f'submitted_s{seed}_inline.json'
        if receipt.exists():
            print('Existing receipt:', receipt.read_text(), flush=True)
            continue
        assert not list((root / 'runs').iterdir()) and not list((root / 'preflight').iterdir())
        assert not (root / 'results/lane_status.json').exists()
        for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
        jobname = f'robotic_e4_s{seed}_inline'
        active = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()),
                                         '--name', jobname, '-o', '%i'], text=True).strip()
        assert not active, 'Existing job without receipt; reconcile before retrying: ' + active
        env['E4_DIAGNOSTIC_OUTPUT'] = str(root / 'results/egl_ready.json')
        wrapper = shlex.join([str(project / 'envs/e4_20261123/bin/python'), str(diagnostic)])
        wrapper += ' && exec ' + shlex.join(['bash', str(root / 'train.sbatch'), str(project), root.name])
        command = ['sbatch', '--parsable', '--account', args.account,
            '--partition=a100_batch,h800_batch', '--time=3-00:00:00', '--export=ALL',
            '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8', '--mem=40G',
            '--job-name', jobname, '--chdir', str(root), '--output', str(root / 'logs/slurm_%j.out'),
            '--wrap', wrapper]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        value = {'job_id': job, 'seed': seed, 'release': root.name, 'time': time.time(),
            'command': command, 'diagnostic_sha256': hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
            'supersedes': ['29411', '29412'], 'scope': 'Corrected render check, both full-model preflights, then training',
            'graphics_environment': {k: env[k] for k in ('LD_LIBRARY_PATH', '__EGL_VENDOR_LIBRARY_FILENAMES')}}
        tmp = receipt.with_suffix('.json.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(receipt)
        print(json.dumps(value), flush=True)
