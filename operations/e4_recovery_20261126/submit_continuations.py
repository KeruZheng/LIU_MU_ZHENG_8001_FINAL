"""Idempotent afterany submissions; quota deferral is reported for retry."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ops = Path(__file__).resolve().parent
project = ops.parents[1]
previous = project / 'operations/e4_parallel_20261123'
graphics = json.loads((previous / 'nvidia-egl-runtime.json').read_text())
libraries = json.loads((previous / 'runtime-libraries.json').read_text())['library_paths']
env = dict(os.environ, MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', WANDB_MODE='disabled',
           OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1',
           LD_LIBRARY_PATH=':'.join([graphics['library_path']]+libraries),
           __EGL_VENDOR_LIBRARY_FILENAMES=graphics['vendor_file'])
results = []
with (ops / 'continuation_submission.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed, old_job in [(1, '29427'), (2, '29428')]:
        root = project / f'releases/e4_continue_s{seed}_20261126_v1'
        receipt = root / 'submission.json'
        if receipt.exists():
            results.append(json.loads(receipt.read_text()))
            continue
        checks = [ops / f'validation_s{seed}/passed.json', ops / 'v3_regression_remote/scores.json']
        if not all(path.exists() for path in checks):
            results.append({'seed': seed, 'state': 'waiting_validation',
                            'reason': 'Actual remote checkpoint and full D2 regression must pass first'})
            continue
        assert json.loads(checks[0].read_text())['state'] == 'passed'
        regression = json.loads(checks[1].read_text())
        assert regression['max_steps'] == 400 and set(regression['scores']) == {
            str(i) for i in range(200012, 200016)}
        for name, expected in json.loads((root / 'frozen_files.json').read_text()).items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
        jobname = f'e4_continue_s{seed}_1126'
        active = subprocess.check_output(['squeue', '-h', '-u', str(os.getuid()),
                                         '--name', jobname, '-o', '%i'], text=True).strip()
        assert not active, 'Submission receipt missing; reconcile active job '+active
        command = [str(project / 'envs/e4_20261123/bin/python'), '-u', str(root / 'lane.py'),
                   '--root', str(root), '--project', str(project)]
        submit = ['sbatch', '--parsable', '--account=qingbiaoli', '--partition=a100_batch,h800_batch',
                  '--time=3-00:00:00', '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8',
                  '--mem=40G', '--export=ALL', '--dependency=afterany:'+old_job,
                  '--job-name='+jobname, '--chdir='+str(root), '--output='+str(root/'logs/slurm_%j.out'),
                  '--wrap', shlex.join(command)]
        run = subprocess.run(submit, env=env, capture_output=True, text=True)
        if run.returncode:
            reason = run.stderr.strip()
            if any(k in reason for k in ['MaxSubmitJobsPerAccount', 'QOSMaxSubmitJobPerUserLimit', 'QOSMaxSubmitJobPerAccountLimit',
                                         'AssocMaxSubmitJobLimit', 'job violates accounting/QOS policy']):
                results.append({'seed': seed, 'state': 'waiting_submission_slot', 'reason': reason})
                continue
            raise RuntimeError(reason)
        job = run.stdout.strip().split(';')[0]
        assert job.isdigit(), run.stdout
        value = {'seed': seed, 'state': 'submitted', 'job_id': job, 'source_job': old_job,
                 'release': root.name, 'submitted_at': time.time(), 'command': submit}
        temp = receipt.with_suffix('.json.tmp')
        temp.write_text(json.dumps(value, indent=2)+'\n')
        temp.replace(receipt)
        results.append(value)
print(json.dumps({'time': time.time(), 'lanes': results}))
