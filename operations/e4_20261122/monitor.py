"""Observe E4 without changing its queue, models or frozen source."""
import argparse
from datetime import datetime
import fcntl
import html
import json
import os
from pathlib import Path
import subprocess
import time


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(value)
    temporary.replace(path)


def tail(path):
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        return stream.read().decode('utf-8', errors='replace').replace('\r', '\n')[-2000:]


def snapshot(root, codes, previous):
    now = time.time()
    supervisor = read_json(root / 'results/supervisor_status.json')
    alerts = []
    terminal = supervisor.get('state') in ('complete', 'failed')
    pid = supervisor.get('supervisor_pid')
    command = Path('/proc') / str(pid) / 'cmdline'
    alive = command.exists() and str(root).encode() in command.read_bytes()
    age = now - supervisor.get('heartbeat', 0)
    if not terminal and (not alive or age > 180):
        alerts.append('Queue owner missing or heartbeat older than 180 seconds')
    if supervisor.get('state') == 'failed':
        alerts.append(supervisor.get('error', 'Queue failed'))
    downloads = {}
    for item in read_json(root / 'data_sources.json')['files']:
        name = Path(item['path']).stem
        raw = codes / f'data/robomimic/datasets/{name}/{name}.hdf5'
        path = raw if raw.exists() else raw.with_suffix('.hdf5.part')
        size = path.stat().st_size if path.exists() else 0
        old = previous.get('downloads', {}).get(name, {})
        elapsed = now - previous.get('timestamp', now)
        speed = (size - old['bytes']) / elapsed if elapsed > 0 and 'bytes' in old else None
        downloads[name] = {'bytes': size, 'expected': item['size'],
            'percent': round(100 * size / item['size'], 1),
            'bytes_per_second': speed,
            'eta_seconds': (item['size'] - size) / speed if speed and speed > 0 else None}
    runs = []
    for seed in (0, 1, 2):
        for method in ('equidiff', 'dp'):
            name = f'{method}_square_d0_s{seed}'
            directory = root / 'runs' / name
            status = read_json(directory / 'completed.json') or read_json(directory / 'status.json')
            scores = {}
            for domain in ('d0', 'd1', 'd2'):
                result = read_json(directory / f'transfer_{domain}/scores.json')
                if result:
                    scores[domain] = {'episodes': result['episodes'],
                        'success': result['scores'].get('test/mean_score')}
            runs.append({'name': name, 'state': status.get('state', 'pending'),
                'epoch': status.get('epoch'), 'global_step': status.get('global_step'),
                'target_updates': status.get('target_updates'), 'scores': scores})
    preflight = {method: read_json(root / f'preflight/{method}_square_d0_s0/result.json').get('state', 'pending')
                 for method in ('equidiff', 'dp')}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15)
    if gpu.returncode:
        alerts.append('nvidia-smi failed: ' + gpu.stderr.strip())
    logs = sorted((root / 'logs').glob('*.log'), key=lambda p: p.stat().st_mtime)
    latest_log = logs[-1] if logs else None
    # Use byte and step counts, not heartbeat timestamps, to detect lack of progress.
    progress = json.dumps([supervisor.get('state'), [d['bytes'] for d in downloads.values()], runs], sort_keys=True)
    changed_at = now if progress != previous.get('progress_key') else previous.get('changed_at', now)
    if not terminal and now - changed_at > 1800:
        alerts.append('No observed byte/epoch/step/phase progress for 30 minutes; inspect logs (evaluation can be slow)')
    return {'timestamp': now, 'updated': datetime.now().astimezone().isoformat(timespec='seconds'),
        'monitor_pid': os.getpid(), 'supervisor': supervisor, 'owner_alive': alive,
        'heartbeat_age_seconds': round(age, 1), 'downloads': downloads, 'preflight': preflight,
        'runs': runs, 'gpu': gpu.stdout.strip(), 'alerts': alerts,
        'progress_key': progress, 'changed_at': changed_at,
        'latest_log': latest_log.name if latest_log else None,
        'log_tail': tail(latest_log) if latest_log else ''}


def render(record):
    owner = record['supervisor']
    lines = ['# E4 live status', '', 'Updated: ' + record['updated'], '',
        f"Local RTX 5090 | phase: {owner.get('state')} | completed: {owner.get('completed_runs', 0)}/6",
        f"Owner alive: {record['owner_alive']} | heartbeat age: {record['heartbeat_age_seconds']} s",
        '', 'GPU (name, utilization %, used MiB, total MiB): ' + record['gpu'], '',
        'Alerts: ' + ('; '.join(record['alerts']) or 'none'), '',
        '| Dataset | Downloaded | Recent MiB/s | Download ETA |', '|---|---:|---:|---:|']
    for name, item in record['downloads'].items():
        speed = item['bytes_per_second']
        speed_text = f'{speed / 1048576:.2f}' if speed is not None else '-'
        eta = item['eta_seconds']
        eta_text = f'{eta / 60:.1f} min' if eta is not None else '-'
        lines.append(f"| {name} | {item['percent']}% | {speed_text} | {eta_text} |")
    lines.extend(['', 'Full download is not hash verification or GPU training.', '',
        'Preflight: ' + json.dumps(record['preflight']), '',
        '| Run | Phase | Epoch / 500 | Updates | Final domain results |', '|---|---|---:|---:|---|'])
    for run in record['runs']:
        scores = json.dumps(run['scores']) if run['scores'] else '-'
        lines.append(f"| {run['name']} | {run['state']} | {run['epoch']} | {run['global_step']} | {scores} |")
    lines.extend(['', 'Latest log: ' + str(record['latest_log']), '', '```text', record['log_tail'], '```', '',
        'Refresh: 30 seconds. This observer does not restart training or send external notifications.',
        'Training ETA will require measured throughput; download ETA excludes conversion and preflight.', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--codes', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(args.output / 'monitor.pid', str(os.getpid()) + '\n')
        previous = read_json(args.output / 'latest.json')
        while True:
            try:
                record = snapshot(args.root.resolve(), args.codes.resolve(), previous)
                markdown = render(record)
                write(args.output / 'latest.json', json.dumps(record, indent=2) + '\n')
                write(args.output / 'STATUS.md', markdown)
                write(args.output / 'status.html', '<!doctype html><meta charset="utf-8">'
                    '<meta http-equiv="refresh" content="30"><title>E4 live status</title>'
                    '<style>body{margin:2em;background:#f4f1e8;color:#182d28}pre{white-space:pre-wrap}</style>'
                    '<pre>' + html.escape(markdown) + '</pre>')
                with (args.output / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                previous = record
                print(record['updated'], record['supervisor'].get('state'), record['alerts'], flush=True)
            except Exception as exc:
                write(args.output / 'monitor_error.json', json.dumps({'time': time.time(), 'error': repr(exc)}))
                print(repr(exc), flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
