"""Read metadata on a cluster login node; no GPU/model/simulator computation."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time


def command(args):
    try:
        p = subprocess.run(args, text=True, capture_output=True, timeout=20)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': -1, 'stderr': str(exc)}


def probe(root, jobs, releases):
    result = {'time_epoch': time.time(), 'jobs': {}, 'releases': {}}
    result['squeue'] = command(['squeue', '-h', '-j', ','.join(jobs), '-o', '%i|%T|%M|%R|%b'])
    result['sacct'] = command(['sacct', '-n', '-X', '-P', '-j', ','.join(jobs),
                               '--format=JobID,State,Elapsed,ExitCode,NodeList'])
    for line in result['sacct'].get('stdout', '').splitlines():
        f = line.split('|')
        if len(f) >= 5 and f[0] in jobs:
            result['jobs'][f[0]] = dict(state=f[1], elapsed=f[2], exit_code=f[3], node=f[4])
    for line in result['squeue'].get('stdout', '').splitlines():
        f = line.split('|')
        if len(f) >= 5 and f[0] in jobs:
            result['jobs'].setdefault(f[0], {}).update(state=f[1], elapsed=f[2], reason_or_node=f[3], gres=f[4])
    disk = shutil.disk_usage(root)
    result['filesystem_free_bytes'] = disk.free
    result['filesystem_is_not_user_quota'] = True
    for release in releases:
        entry = {}
        for name in ('supervisor_status', 'lane_status', 'preflight'):
            p = root / 'releases' / release / 'results' / (name + '.json')
            if p.exists():
                try:
                    entry[name] = json.loads(p.read_text())
                    entry[name + '_age_seconds'] = max(0, time.time() - p.stat().st_mtime)
                except (OSError, ValueError) as exc:
                    entry[name + '_read_error'] = str(exc)
        result['releases'][release] = entry
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--jobs', required=True)
    p.add_argument('--releases', required=True)
    a = p.parse_args()
    print(json.dumps(probe(a.root, a.jobs.split(','), a.releases.split(','))))
