"""Retry quota-deferred submissions and mirror new-lane evidence without models."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from evaluate import atomic

p = argparse.ArgumentParser()
p.add_argument('--workspace', type=Path, required=True)
a = p.parse_args()
out = a.workspace / 'codes/tasks/rq1_transfer/results/e4_recovery_20261126'
out.mkdir(parents=True, exist_ok=True)
cluster = json.loads((a.workspace / 'codes/configs/clusters.local.json').read_text())['sicc']
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', *cluster['ssh_options'], cluster['host']]
remote = cluster['root']
with (out / 'watch.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        state = {'pid': os.getpid(), 'time': time.time(), 'lanes': [], 'errors': []}
        try:
            value = subprocess.check_output(ssh + ['python3 '+remote+'/operations/e4_recovery_20261126/submit_continuations.py'],
                                            text=True, timeout=90)
            state.update(json.loads(value))
            queue_command = shlex.join(['squeue', '-u', cluster['host'].split('@')[0],
                                        '-h', '-o', '%i %j %T %M %R'])
            state['queue'] = subprocess.check_output(ssh + [queue_command], text=True, timeout=30)
            for seed in [1, 2]:
                release = f'e4_continue_s{seed}_20261126_v1'
                target = out / release
                target.mkdir(exist_ok=True)
                for directory in ['results', 'runs', 'logs']:
                    subprocess.run(['rsync', '-a', '--partial', '--exclude=checkpoints/',
                                    '--exclude=final_ema.pt', '--exclude=wandb/', '--exclude=media/',
                                    '-e', shlex.join(ssh[:-1]), ssh[-1]+':'+remote+'/releases/'+release+'/'+directory+'/',
                                    str(target / directory)+'/'], check=True, timeout=90,
                                   stdout=subprocess.DEVNULL)
                path = target / 'results/lane_status.json'
                if path.exists():
                    state.setdefault('runtime', {})[str(seed)] = json.loads(path.read_text())
            local = a.workspace / 'codes/tasks/rq1_transfer/results/e4_corrected_eval_20261126_v3/attempt2/pair_status.json'
            if local.exists():
                state['local_corrected_pair'] = json.loads(local.read_text())
        except Exception as exc:
            state['errors'].append(repr(exc))
        atomic(out / 'latest.json', state)
        with (out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(state)+'\n')
        lines = ['# E4 recovery monitor', '', 'Updated Unix time: '+str(state['time']), '',
                 'This observer does not stop training or edit frozen sources.', '']
        for lane in state.get('lanes', []):
            lines.append(f"- Seed {lane['seed']}: {lane['state']}; job {lane.get('job_id', 'not yet allocated a submission slot')}.")
        for seed, lane in state.get('runtime', {}).items():
            lines.append(f"- Runtime seed {seed}: {lane['state']}, {lane.get('run', '')}.")
        lines += ['', 'Errors: '+json.dumps(state['errors']), '', 'See latest.json for scheduler and evaluation details.']
        (out / 'STATUS.md').write_text('\n'.join(lines)+'\n')
        time.sleep(60)
