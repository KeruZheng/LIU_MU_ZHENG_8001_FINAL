"""Require affirmative scheduler evidence before using an old checkpoint."""
import os
import subprocess


def require_terminal(job_id, runner=subprocess.run):
    job_id = str(job_id)
    if not job_id.isdigit():
        raise ValueError('Invalid source job ID')
    # A retired ID may make `squeue -j` fail. Query the user's active jobs instead.
    queue = runner(['squeue', '-h', '-u', str(os.getuid()), '-o', '%i'],
                   text=True, capture_output=True, check=True, timeout=30)
    if job_id in queue.stdout.split():
        raise RuntimeError('Source job is still active: ' + job_id)
    account = runner(['sacct', '-n', '-X', '-j', job_id, '--format=JobIDRaw,State', '-P'],
                     text=True, capture_output=True, check=True, timeout=30)
    states = []
    for line in account.stdout.splitlines():
        fields = line.strip().split('|')
        if len(fields) >= 2 and fields[0] == job_id and fields[1].strip():
            states.append(fields[1].split()[0].rstrip('+'))
    terminal = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE', 'PREEMPTED', 'REVOKED'}
    if len(states) != 1 or states[0] not in terminal:
        raise RuntimeError('No unambiguous terminal accounting record for ' + job_id
                           + ': ' + repr(states))
    return {'source_job': job_id, 'scheduler_state': states[0]}
