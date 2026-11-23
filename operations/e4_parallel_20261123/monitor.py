"""Combined local/SICC E4 observer; remote evidence comes from the fleet collector."""
import argparse
from datetime import datetime
import fcntl
import html
import json
import os
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def atomic(path, text):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(text)
    temp.replace(path)


def observe(codes, root):
    fleet = read(codes / 'operations/fleet/results/latest.json')
    alerts, runs, queues = [], [], []
    now = time.time()
    jobs = {item.get('release'): item for item in fleet.get('auxiliary', [])}
    for seed in (0, 1, 2):
        release = root.name if seed == 0 else f'e4_square_d0_s{seed}_sicc_20261123_v1'
        evidence = root if seed == 0 else codes / 'tasks/rq1_transfer/results' / release
        queue = read(evidence / 'results' / ('supervisor_status.json' if seed == 0 else 'lane_status.json'))
        job = jobs.get(release, {})
        scheduler = 'LOCAL' if seed == 0 else job.get('scheduler', 'UNKNOWN')
        if seed == 0:
            try:
                command = (Path('/proc') / str(queue['supervisor_pid']) / 'cmdline').read_bytes()
                alive = b'local_owner.py' in command
            except (OSError, KeyError):
                alive = False
            if queue.get('state') not in ('complete', 'failed') and (not alive or now - queue.get('heartbeat', 0) > 180):
                alerts.append('Local E4 owner missing or stale')
        elif now - fleet.get('time_epoch', 0) > 600:
            alerts.append(f'Seed {seed}: fleet evidence older than 10 minutes')
        if queue.get('state') == 'failed' or scheduler in ('FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'):
            alerts.append(f'Seed {seed}: {scheduler}; {queue.get("error", queue.get("state"))}')
        queues.append({'seed': seed, 'device': '5090' if seed == 0 else 'SICC',
            'release': release, 'scheduler': scheduler, 'job_id': job.get('job_id'),
            'phase': queue.get('state', 'not_started'), 'completed_runs': queue.get('completed_runs', 0),
            'evidence_updated': fleet.get('time') if seed else datetime.now().astimezone().isoformat()})
        for method in ('equidiff', 'dp'):
            name = f'{method}_square_d0_s{seed}'
            output = evidence / 'runs' / name
            complete = read(output / 'completed.json')
            status = complete or read(output / 'status.json')
            scores = {domain: read(output / f'transfer_{domain}/scores.json') for domain in ('d0', 'd1', 'd2')}
            done = complete.get('state') == 'complete' and all(v.get('episodes') == 100 for v in scores.values())
            runs.append({'name': name, 'device': '5090' if seed == 0 else 'SICC',
                'state': 'complete' if done else status.get('state', 'not_started'),
                'epoch': status.get('epoch'), 'step': status.get('global_step'),
                'complete': done, 'scores': {d: v.get('scores', {}).get('test/mean_score') for d, v in scores.items()}})
    return {'updated': datetime.now().astimezone().isoformat(timespec='seconds'), 'timestamp': now,
            'monitor_pid': os.getpid(), 'queues': queues, 'runs': runs, 'alerts': alerts,
            'completed_runs': sum(r['complete'] for r in runs), 'planned_runs': 6}


def render(record):
    lines = ['# E4 live status: local 5090 + SICC', '', 'Updated: ' + record['updated'], '',
             f"Complete training + all final evaluations: {record['completed_runs']}/6", '',
             'Alerts: ' + ('; '.join(record['alerts']) or 'none'), '',
             '| Seed | Device | Job | Scheduler | Queue phase |', '|---|---|---|---|---|']
    for q in record['queues']:
        lines.append(f"| {q['seed']} | {q['device']} | {q['job_id'] or '-'} | {q['scheduler']} | {q['phase']} |")
    lines += ['', '| Run | Device | Phase | Epoch / 500 | Updates / 112000 | D0 / D1 / D2 |',
              '|---|---|---|---:|---:|---|']
    for r in record['runs']:
        scores = ' / '.join('-' if v is None else str(v) for v in r['scores'].values())
        lines.append(f"| {r['name']} | {r['device']} | {r['state']} | {r['epoch']} | {r['step']} | {scores} |")
    lines += ['', 'Local refresh: 30 seconds. SICC evidence: fleet collector, about 2 minutes.',
              'PENDING is waiting for resources, not training. Missing final scores are not zero scores.',
              'Remote models are copied locally and SHA256-verified before cleanup and next-method dispatch.',
              'No external notifications or automatic training retries.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', type=Path, required=True)
    parser.add_argument('--local-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic(args.output / 'monitor.pid', str(os.getpid()) + '\n')
        while True:
            try:
                record = observe(args.codes.resolve(), args.local_root.resolve())
                markdown = render(record)
                atomic(args.output / 'latest.json', json.dumps(record, indent=2) + '\n')
                atomic(args.output / 'STATUS.md', markdown)
                atomic(args.output / 'status.html', '<!doctype html><meta charset="utf-8">'
                    '<meta http-equiv="refresh" content="30"><title>E4 live status</title>'
                    '<style>body{margin:2em;background:#f4f1e8;color:#182d28}pre{white-space:pre-wrap}</style>'
                    '<pre>' + html.escape(markdown) + '</pre>')
                with (args.output / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(record['updated'], record['completed_runs'], record['alerts'], flush=True)
            except Exception as exc:
                atomic(args.output / 'monitor_error.json', json.dumps({'time': time.time(), 'error': repr(exc)}))
                print(repr(exc), flush=True)
                if args.once:
                    raise
            if args.once:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
