"""Independent fleet monitor and task-owned evidence copies. Never dispatches jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

FAILURES = {'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL'}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def execute(args, timeout=60, text=None):
    try:
        p = subprocess.run(args, input=text, text=True, capture_output=True, timeout=timeout)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr[-3000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': -1, 'stderr': str(exc)}


def check_process(pid, expected):
    try:
        command = Path('/proc/%s/cmdline' % int(pid)).read_bytes().replace(b'\0', b' ').decode()
        return expected in command
    except (OSError, ValueError, TypeError):
        return False


def sync_metadata(entry, server, codes):
    target = codes / 'tasks' / entry['owner'] / 'results' / entry['release']
    target.mkdir(parents=True, exist_ok=True)
    args = ['rsync', '-a', '--no-owner', '--no-group', '--exclude=*.tmp', '--exclude=*.pt',
            '--exclude=*.ckpt', '--include=*/', '--include=*.json', '--include=*.jsonl', '--exclude=*']
    if server:
        args += ['-e', shlex.join(server['ssh'][:-1])]
        source = server['ssh'][-1] + ':' + server['root'] + '/releases/' + entry['release']
    else:
        source = entry['root']
    # Only these trees: never traverse datasets, environments, source symlinks or quarantine.
    result = execute(args + [source + '/runs', source + '/results', str(target) + '/'], timeout=90)
    atomic(target / 'origin.local.json', {'source': source, 'release': entry['release'],
        'device': entry['device'], 'observed_at': datetime.now(timezone.utc).isoformat(),
        'sync_returncode': result['returncode'], 'checkpoints_copied': False})
    return result


def classify(scheduler, training, reports, milestones, split, sync_ok=True):
    if not sync_ok:
        return 'unknown_stale_evidence'
    done = all(reports.get(s, {}).get('state') == 'complete' and
               reports[s].get('summary', {}).get('episodes') == 100 for s in milestones)
    if done and training.get('state') == 'complete' and training.get('step', 0) >= max(milestones):
        return 'complete'
    if scheduler.split('+')[0].split(' ')[0] in FAILURES:
        return 'failed_' + scheduler.lower()
    if scheduler in ('PENDING', 'CONFIGURING', 'SUSPENDED'):
        return scheduler.lower()
    if training.get('state') == 'complete':
        return 'evaluating_or_awaiting_next_milestone'
    if training.get('state') == 'training':
        return 'training' if scheduler in ('RUNNING', 'LOCAL_RUNNING') else 'interrupted_or_unknown'
    return 'not_started'


def progress_alert(key, token, previous, now, running, threshold=1800):
    old = previous.get(key, {})
    changed_at = now if old.get('token') != token else old.get('changed_at', now)
    state = {'token': token, 'changed_at': changed_at}
    return state, running and now - changed_at > threshold


def collect(config, out):
    now = time.time()
    codes = Path(config['codes'])
    prior = read(out / 'latest.json', {})
    state = {'time': datetime.now(timezone.utc).isoformat(), 'time_epoch': now,
             'devices': {}, 'lanes': [], 'services': [], 'alerts': [], 'progress_memory': {}}
    servers = {s['name']: s for s in config['servers']}
    probe_source = Path(__file__).with_name('probe.py').read_text()

    def remote(server):
        releases = [e['release'] for e in config['lanes'] if e['device'] == server['name']]
        jobs = sorted(set([str(e['job_id']) for e in config['lanes'] if e['device'] == server['name']] + server.get('extra_jobs', [])))
        cmd = ['python3', '-', '--root', server['root'], '--jobs', ','.join(jobs), '--releases', ','.join(releases)]
        result = execute(server['ssh'] + [shlex.join(cmd)], timeout=55, text=probe_source)
        if result['returncode'] == 0:
            try:
                return server['name'], json.loads(result['stdout'])
            except ValueError:
                pass
        return server['name'], {'error': result}

    with ThreadPoolExecutor(max_workers=2) as pool:
        for name, result in pool.map(remote, servers.values()):
            state['devices'][name] = result
            if 'error' in result:
                state['alerts'].append(name + ': SSH/probe failed; status unknown')
    state['devices']['local5090'] = execute(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], timeout=15)
    # Sequential evidence sync avoids duplicating large concurrent filesystem reads.
    syncs = {}
    for entry in config['lanes']:
        name = entry['release']
        syncs[name] = sync_metadata(entry, servers.get(entry['device']), codes)
        if syncs[name]['returncode']:
            state['alerts'].append(name + ': evidence sync failed; old copies are not live evidence')
    indices = {task: [] for task in config['tasks']}
    for entry in config['lanes']:
        release = entry['release']
        snapshot = codes / 'experiments/releases' / release
        evidence = codes / 'tasks' / entry['owner'] / 'results' / release
        if entry['device'] == 'local5090':
            active = read(Path(entry['root']) / 'results/supervisor_status.json', {})
            alive = check_process(active.get('supervisor_pid'), 'supervise.py')
            scheduler = 'LOCAL_RUNNING' if alive else 'LOCAL_STOPPED'
            queue = active
        else:
            device = state['devices'][entry['device']]
            scheduler = device.get('jobs', {}).get(str(entry['job_id']), {}).get('state', 'UNKNOWN')
            records = device.get('releases', {}).get(release, {})
            queue = records.get('supervisor_status') or records.get('lane_status', {})
            if 'error' in device:
                scheduler = 'UNKNOWN'
        fresh = syncs[release]['returncode'] == 0 and scheduler != 'UNKNOWN'
        counts = {'complete': 0, 'planned': len(entry['runs'])}
        active_rows = []
        for run in entry['runs']:
            run_dir = evidence / 'runs' / run
            training = read(run_dir / 'status.json', {})
            reports = {s: read(run_dir / (entry['split'] + '_step_%s.json' % s), {}) for s in entry['milestones']}
            status = classify(scheduler, training, reports, entry['milestones'], entry['split'], fresh)
            counts['complete'] += status == 'complete'
            runtime = read(run_dir / 'runtime.json', {})
            files = {}
            # Small JSON evidence is content-addressed; changing training reports remain snapshots.
            for p in sorted(run_dir.glob('*.json')):
                raw = p.read_bytes()
                files[p.name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            checkpoints = []
            for source in entry.get('checkpoint_sources', []):
                directory = Path(source) / run
                models = sorted(p.name for p in directory.glob('*.pt')) if directory.is_dir() else []
                if models:
                    checkpoints.append({'directory': str(directory), 'files': models,
                                        'verification': 'Consult the original archive receipts; existence alone is not verification'})
            if checkpoints:
                link = codes / 'tasks' / entry['owner'] / 'results/checkpoints' / release / run
                link.parent.mkdir(parents=True, exist_ok=True)
                if not link.exists() and not link.is_symlink():
                    link.symlink_to(checkpoints[0]['directory'], target_is_directory=True)
            config_files = sorted(str(p.relative_to(codes)) for p in (snapshot / 'configs').glob(run + '*.json')
                                  if p.stem == run or p.stem.startswith(run + '_'))
            row = {'run_id': release + '/' + run, 'release': release, 'run': run,
                   'study': entry['study'], 'state': status, 'device': entry['device'],
                   'job_id': entry.get('job_id'), 'scheduler_state': scheduler,
                   'training_step': training.get('step'), 'runtime_gpu': runtime.get('gpu'),
                   'torch': runtime.get('torch'), 'split': entry['split'],
                   'evidence': str(run_dir.relative_to(codes)),
                   'code_snapshot': str(snapshot.relative_to(codes)),
                   'snapshot_sha256': hashlib.sha256((snapshot / 'SNAPSHOT.json').read_bytes()).hexdigest(),
                   'configuration_files': config_files, 'available_local_checkpoints': checkpoints,
                   'checkpoint_sources': entry.get('checkpoint_sources', []),
                   'files': files, 'scores': {str(s): {
                       'state': r.get('state'), 'episodes': r.get('summary', {}).get('episodes'),
                       'mean_max_coverage': r.get('summary', {}).get('mean_max_coverage')}
                       for s, r in reports.items()}}
            for task in entry['tasks']:
                indices[task].append(row)
            if status not in ('not_started', 'complete', 'pending'):
                active_rows.append([run, training.get('step'), [(s, r.get('summary', {}).get('episodes')) for s, r in reports.items()]])
        token = json.dumps([queue.get('name'), queue.get('run'), queue.get('state'), active_rows], sort_keys=True)
        memory, stalled = progress_alert(release, token, prior.get('progress_memory', {}), now,
                                         scheduler in ('RUNNING', 'LOCAL_RUNNING'))
        state['progress_memory'][release] = memory
        if stalled:
            state['alerts'].append(release + ': no observed step/episode/phase progress for 30 minutes')
        if scheduler.split('+')[0].split(' ')[0] in FAILURES:
            state['alerts'].append(release + ': Slurm ' + scheduler)
        if scheduler == 'LOCAL_STOPPED' and counts['complete'] < counts['planned']:
            state['alerts'].append(release + ': local supervisor not running with unfinished work')
        state['lanes'].append(dict(release=release, device=entry['device'], scheduler=scheduler,
            current=queue.get('name') or queue.get('run') or '-', phase=queue.get('state'),
            progress=queue.get('progress', {}), job_id=entry.get('job_id'), counts=counts))
    for service in config['services']:
        try:
            pid = Path(service['pid_file']).read_text().strip()
        except OSError:
            pid = None
        age = None
        p = Path(service['heartbeat'])
        if p.exists():
            age = max(0, now - p.stat().st_mtime)
        alive = check_process(pid, service['command_contains'])
        row = {'name': service['name'], 'pid': pid, 'alive': alive, 'heartbeat_age_seconds': age}
        state['services'].append(row)
        if not alive or age is None or age > service.get('max_age', 600):
            state['alerts'].append(service['name'] + ': process absent or heartbeat stale')
        status = read(p, {})
        if status.get('errors'):
            state['alerts'].append(service['name'] + ': reported errors; inspect its status file')
    state['local_disk_free_bytes'] = shutil.disk_usage(codes).free
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable'):
            memory[key] = int(value.strip().split()[0]) * 1024
    state['local_memory_bytes'] = memory
    if state['local_disk_free_bytes'] < 100 * 1024 ** 3:
        state['alerts'].append('Local archive filesystem has less than 100 GiB free')
    for task, rows in indices.items():
        atomic(codes / 'tasks' / task / 'results/run_index.json', {'time': state['time'], 'runs': rows,
            'deduplication': 'run_id is global; RQ1/RQ2 share main runs; never add their counts together'})
        public = [{k: r[k] for k in ('run_id', 'study', 'state', 'device', 'training_step', 'runtime_gpu', 'torch', 'split', 'evidence', 'code_snapshot', 'snapshot_sha256', 'scores')} for r in rows]
        atomic(codes / 'tasks' / task / 'reports/latest.json', {'time': state['time'], 'runs': public})
    state['sync'] = syncs
    atomic(out / 'latest.json', state)
    with (out / 'history.jsonl').open('a') as stream:
        stream.write(json.dumps(state) + '\n')
    lines = ['# Fleet status', '', 'Observed: ' + state['time'], '',
             'PENDING means queued, not computing. Counts include all required evaluations.', '',
             '| Device | Release | Job | Scheduler | Current operation | Complete |',
             '|---|---|---|---|---|---|']
    for lane in state['lanes']:
        lines.append('| {device} | {release} | {job_id} | {scheduler} | {current} | {complete}/{planned} |'.format(**lane, **lane['counts']))
    lines += ['', '## Alerts', ''] + (['- ' + a for a in state['alerts']] or ['No current alerts.'])
    lines += ['', '## Monitoring services', '']
    lines += ['- %s: alive=%s, heartbeat age=%s seconds' % (s['name'], s['alive'], s['heartbeat_age_seconds']) for s in state['services']]
    lines += ['', 'Checkpoint copies remain in existing verified archives; task indices record their locations.',
              'Filesystem free space is not account quota. No training is stopped or resubmitted by this monitor.', '']
    p = out / 'STATUS.md'
    temp = p.with_suffix('.tmp'); temp.write_text('\n'.join(lines)); temp.replace(p)
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--interval', type=int, default=120)
    p.add_argument('--once', action='store_true')
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    with (a.output / 'collector.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (a.output / 'collector.pid').write_text(str(os.getpid()) + '\n')
        while True:
            atomic(a.output / 'heartbeat.json', {'time_epoch': time.time(), 'phase': 'collecting'})
            try:
                state = collect(read(a.config), a.output)
                atomic(a.output / 'collector_error.json', {'time_epoch': time.time(), 'error': None})
                print(json.dumps({'time': state['time'], 'alerts': state['alerts']}), flush=True)
                atomic(a.output / 'heartbeat.json', {'time_epoch': time.time(), 'phase': 'idle'})
            except Exception as exc:
                atomic(a.output / 'collector_error.json', {'time_epoch': time.time(), 'error': repr(exc)})
                if a.once:
                    raise
                print(repr(exc), flush=True)
            if a.once:
                break
            time.sleep(a.interval)


if __name__ == '__main__':
    main()
