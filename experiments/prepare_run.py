"""Materialize an isolated, auditable reproduction under its research task."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil

CODES = Path(__file__).resolve().parents[1]


def prepare(task, release, run, label, data, upstream):
    for value in (release, run, label):
        if Path(value).name != value or value in ('', '.', '..'):
            raise ValueError('Expected a single path component')
    plan = json.loads((CODES / 'tasks' / task / 'task.json').read_text())
    if release not in plan['releases']:
        raise ValueError('Release does not belong to this task')
    source = CODES / 'experiments/releases' / release
    manifest = json.loads((source / 'SNAPSHOT.json').read_text())
    if not (source / 'configs' / (run + '.json')).is_file():
        raise ValueError('Unknown run configuration')
    for relative, expected in manifest['files'].items():
        if hashlib.sha256((source / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed release file: ' + relative)
    data, upstream = data.resolve(strict=True), upstream.resolve(strict=True)
    target = CODES / 'tasks' / task / 'runs' / label
    target.mkdir(parents=True, exist_ok=False)
    for relative in list(manifest['files']) + ['SNAPSHOT.json']:
        p = target / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, p)
    (target / 'data').symlink_to(data, target_is_directory=True)
    (target / 'upstream').symlink_to(upstream, target_is_directory=True)
    (target / 'results').mkdir()
    (target / 'RUN_ORIGIN.json').write_text(json.dumps({
        'task': task, 'release': release, 'configuration': run,
        'snapshot_sha256': hashlib.sha256((source / 'SNAPSHOT.json').read_bytes()).hexdigest()
    }, indent=2) + '\n')
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['rq1_transfer', 'rq2_exact_aug', 'rq3_symmetry_breaking'])
    p.add_argument('--release', required=True)
    p.add_argument('--run', required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--data', type=Path, default=CODES / 'data')
    p.add_argument('--upstream', type=Path, default=CODES / 'upstream')
    a = p.parse_args()
    root = prepare(a.task, a.release, a.run, a.label, a.data, a.upstream)
    print('Prepared only; no training launched.')
    print('Working directory:', root)
    print('With a compatible project Python and an allocated GPU:')
    print('PYTHONPATH=src python scripts/validate_c4.py --output results/native_validation.json')
    print('PYTHONPATH=src python -m pusht_study.train --config ' +
          shlex.quote('configs/' + a.run + '.json') + ' --output runs/' + shlex.quote(a.run) + ' --device cuda')
