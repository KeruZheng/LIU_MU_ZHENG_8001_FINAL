"""Submit two registered E4 lanes once, persisting each scheduler receipt."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, required=True)
parser.add_argument('--account', required=True)
args = parser.parse_args()
project = args.project.resolve()
work = project / 'operations/e4_parallel_20261123'
assert json.loads((work / 'imports.json').read_text())['state'] == 'imports_passed'
libraries = json.loads((work / 'runtime-libraries.json').read_text())['library_paths']
env = dict(os.environ, LD_LIBRARY_PATH=':'.join(libraries))
with (work / 'submission.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed in (1, 2):
        name = f'e4_square_d0_s{seed}_sicc_20261123_v1'
        root = project / 'releases' / name
        receipt = work / f'submitted_s{seed}.json'
        if receipt.exists():
            print(receipt.read_text(), flush=True)
            continue
        registered = json.loads((project / 'operations/parallel_20261119/registry.json').read_text())['releases']
        assert name in registered
        for rel, expected in json.loads((root / 'frozen_files.json').read_text()).items():
            assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == expected, rel
        jobname = f'robotic_e4_s{seed}'
        active = subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(),
                                         '--name', jobname, '-o', '%i'], text=True).strip()
        assert not active, f'Existing {jobname}: {active}; reconcile receipt before resubmission'
        command = ['sbatch', '--parsable', '--account', args.account,
            '--partition=a100_batch,h800_batch', '--time=3-00:00:00', '--export=ALL', '--job-name', jobname,
            '--chdir', str(root), '--output', str(root / 'logs/slurm_%j.out'),
            str(root / 'train.sbatch'), str(project), name]
        result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
        job = result.stdout.strip().split(';')[0]
        assert job.isdigit(), result.stdout
        value = {'job_id': job, 'seed': seed, 'release': name, 'submitted_at': time.time(),
                 'command': command, 'LD_LIBRARY_PATH': env['LD_LIBRARY_PATH'],
                 'gpu_preflight_required': True}
        tmp = receipt.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(value, indent=2) + '\n')
        tmp.replace(receipt)
        print(json.dumps(value), flush=True)
