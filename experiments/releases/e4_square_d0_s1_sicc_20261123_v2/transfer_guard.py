"""Verified result offloading for independent lanes on two clusters."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

OP = 'operations/parallel_20261119'
ROOTS = ('/data/qingbiaoli/yc67825/robotic', '/home/user/yc67825/keru/robotic')


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def locate(project, release, run):
    registry = json.loads((project / OP / 'registry.json').read_text())
    if release not in registry['releases'] or Path(run).name != run or run in ('', '.', '..'):
        raise ValueError('Unregistered release/run')
    directory = project / 'releases' / release / 'runs' / run
    if directory.is_symlink() or directory.resolve().parent != (project / 'releases' / release / 'runs').resolve():
        raise ValueError('Invalid run path')
    return directory, release + '__' + run


def checked_path(directory, name):
    rel = Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:
        raise ValueError('Invalid artifact path')
    path = directory / rel
    if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError('Artifact escapes run')
    return path


def ready(directory):
    record = json.loads((directory / 'archive_ready.json').read_text())
    if record.get('state') != 'complete' or not record.get('checkpoints'):
        raise ValueError('Run not ready for offloading')
    for name in record['checkpoints']:
        if checked_path(directory, name).suffix not in ('.pt', '.ckpt'):
            raise ValueError('Only model checkpoints may be removed')
    return record


def inventory(project, release, run):
    directory, key = locate(project, release, run)
    record = ready(directory)
    files = []
    for path in sorted(directory.rglob('*')):
        if not path.is_file() or path.is_symlink() or path.suffix in ('.tmp', '.lock'):
            continue
        files.append({'name': str(path.relative_to(directory)), 'size': path.stat().st_size, 'sha256': sha(path)})
    return {'release': release, 'run': run, 'key': key, 'files': files,
            'checkpoints': record['checkpoints']}


def finalize(project, request):
    directory, key = locate(project, request['release'], request['run'])
    receipt = project / OP / 'receipts' / (key + '.json')
    if receipt.exists():
        return json.loads(receipt.read_text())
    transaction = project / OP / 'transactions' / (key + '.json')
    records = {r['name']: r for r in request['files']}
    if len(records) != len(request['files']):
        raise ValueError('Duplicate artifact')
    with ExitStack() as stack:
        lock_path = directory / 'compute.lock'
        if lock_path.exists():
            lock = stack.enter_context(lock_path.open('r+b'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = ready(directory)
        if record['checkpoints'] != request['checkpoints'] or not set(request['checkpoints']).issubset(records):
            raise ValueError('Missing checkpoint verification')
        previous = json.loads(transaction.read_text()) if transaction.exists() else None
        if previous and (previous['files'] != request['files'] or previous['checkpoints'] != request['checkpoints']):
            raise ValueError('Changed transaction')
        for name, item in records.items():
            path = checked_path(directory, name)
            if not path.exists():
                if previous and name in request['checkpoints']:
                    continue
                raise ValueError('Missing source file')
            if path.stat().st_size != item['size'] or sha(path) != item['sha256']:
                raise ValueError('Checksum mismatch: ' + name)
        atomic(transaction, request)
        freed = 0
        for name in request['checkpoints']:
            path = checked_path(directory, name)
            if path.exists():
                freed += path.stat().st_size; path.unlink()
        result = dict(request, state='archived', time=now(), freed_bytes_this_attempt=freed)
        atomic(receipt, result)
        return result


def gate(project, release, run):
    directory, key = locate(project, release, run)
    receipt = project / OP / 'receipts' / (key + '.json')
    atomic(project / OP / 'requests' / (key + '.json'), {'release': release, 'run': run})
    start = time.monotonic()
    while not receipt.exists():
        if time.monotonic() - start > 12 * 3600:
            raise TimeoutError('No verified archive receipt; next run not dispatched')
        time.sleep(15)
    value = json.loads(receipt.read_text())
    if value.get('state') != 'archived' or value['release'] != release or value['run'] != run:
        raise ValueError('Wrong archive receipt')
    if any((directory / p).exists() for p in value['checkpoints']):
        raise ValueError('Checkpoint cleanup incomplete')


def rpc(server, action, payload=None, request=None):
    command = ['python3', server['root'] + '/' + OP + '/transfer_guard.py', action, '--project', server['root']]
    if request:
        command += ['--release', request['release'], '--run', request['run']]
    p = subprocess.run(server['ssh'] + [shlex.join(command)], input=json.dumps(payload) if payload else '',
                       text=True, capture_output=True, timeout=900)
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return json.loads(p.stdout)


def archive(server, request, output):
    snapshot = rpc(server, 'inventory', request=request)
    target = output / 'archive' / server['name'] / snapshot['release'] / 'runs' / snapshot['run']
    target.mkdir(parents=True, exist_ok=True)
    proof = output / 'proofs' / server['name'] / (snapshot['key'] + '.json')
    records = {r['name']: r for r in snapshot['files']}
    if proof.exists():
        prior = json.loads(proof.read_text())
        for row in prior['files']:
            if row['name'] in snapshot['checkpoints'] and row['name'] not in records:
                snapshot['files'].append(row)
    snapshot['files'].sort(key=lambda r: r['name'])
    transport = shlex.join(server['ssh'][:-1])
    source = server['ssh'][-1] + ':' + server['root'] + '/releases/' + snapshot['release'] + '/runs/' + snapshot['run'] + '/'
    subprocess.run(['rsync', '-a', '--no-owner', '--no-group', '--partial', '--partial-dir=.rsync-partial',
                    '--exclude=*.lock', '--exclude=*.tmp', '-e', transport, source, str(target) + '/'], check=True)
    for row in snapshot['files']:
        path = checked_path(target, row['name'])
        if path.stat().st_size != row['size'] or sha(path) != row['sha256']:
            raise ValueError('Local verification failed: ' + str(path))
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
    snapshot['local_backup'] = str(target)
    atomic(proof, snapshot)
    result = rpc(server, 'finalize', payload=snapshot)
    atomic(output / 'receipts' / server['name'] / (snapshot['key'] + '.json'), result)
    return result


def worker(registry, output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            state = {'time': now(), 'servers': {}, 'errors': []}
            for server in json.loads(registry.read_text())['servers']:
                try:
                    info = rpc(server, 'status')
                    state['servers'][server['name']] = info
                    atomic(output / 'status.json', state)
                    for request in info['requests']:
                        state['active_archive'] = dict(request, cluster=server['name'])
                        atomic(output / 'status.json', state)
                        archive(server, request, output)
                        state.pop('active_archive', None)
                except Exception as exc:
                    state['errors'].append({'cluster': server['name'], 'error': str(exc)})
            atomic(output / 'status.json', state)
            with (output / 'history.jsonl').open('a') as stream:
                stream.write(json.dumps(state) + '\n')
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['inventory', 'finalize', 'status', 'worker'])
    parser.add_argument('--project', choices=ROOTS)
    parser.add_argument('--release'); parser.add_argument('--run')
    parser.add_argument('--registry', type=Path); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.action == 'worker':
        return worker(args.registry, args.output)
    project = Path(args.project)
    if args.action == 'inventory':
        result = inventory(project, args.release, args.run)
    elif args.action == 'finalize':
        result = finalize(project, json.load(__import__('sys').stdin))
    else:
        registry = json.loads((project / OP / 'registry.json').read_text())
        result = {'requests': [], 'lanes': {}}
        for p in sorted((project / OP / 'requests').glob('*.json')):
            if not (project / OP / 'receipts' / p.name).exists():
                result['requests'].append(json.loads(p.read_text()))
        for release in registry['releases']:
            p = project / 'releases' / release / 'results/lane_status.json'
            if p.exists():
                result['lanes'][release] = json.loads(p.read_text())
    print(json.dumps(result))


if __name__ == '__main__':
    main()
