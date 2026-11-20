"""Freeze independent follow-up curves; never change primary queues."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def build(base, method, demos, seeds):
    source = base / (f'conv_dp_n{demos}_20261119_v1' if method == 'dp_c4_aug'
                     else f'conv_c4_n{demos}_20261119_v2')
    name = f'conv_aug_n{demos}_20261120_v1' if method == 'dp_c4_aug' else f'conv_c4_n{demos}_s12_20261120_v1'
    root = base / name
    root.mkdir(exist_ok=False)
    frozen = json.loads((source / 'manifests/frozen_files.json').read_text())
    for relative, expected in frozen.items():
        p = source / relative
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, str(p)
    for directory in ('src', 'scripts', 'manifests'):
        shutil.copytree(source / directory, root / directory,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'frozen_files.json'))
    for directory in ('configs', 'runs', 'logs', 'results'):
        (root / directory).mkdir()
    config = json.loads(next((source / 'configs').glob('*_60000.json')).read_text())
    runs = []
    for seed in seeds:
        run = f'{method}_n{demos}_s{seed}'
        runs.append(run)
        for step in (20000, 40000, 60000):
            write(root / 'configs' / f'{run}_{step}.json', dict(config, method=method, seed=seed, steps=step))
    lane = json.loads((source / 'manifests/lane.json').read_text())
    lane.update(runs=runs, scope='Post-primary-result convergence diagnostic; validation only; no primary test replacement')
    write(root / 'manifests/lane.json', lane)
    (root / 'PROTOCOL.md').write_text(
        f'# Follow-up convergence: {method}, {demos} demos\n\n'
        f'Training seeds: {seeds}; fresh independent 20k/40k/60k curves. '
        'Same split, architecture, optimization and validation seeds as the existing curves. '
        'Only the method (for augmentation) or additional seeds (for C4) change. '
        'This follow-up was designed after observing primary results, so it is exploratory. '
        'No reserved test evaluation or replacement of primary results. '
        'Completed runs require verified local offloading before the next seed.\n')
    for directory in ('data', 'upstream'):
        (root / directory).symlink_to(Path('../..') / directory, target_is_directory=True)
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for directory in ('src', 'scripts', 'manifests', 'configs')
             for p in (root / directory).rglob('*') if p.is_file()}
    files['PROTOCOL.md'] = hashlib.sha256((root / 'PROTOCOL.md').read_bytes()).hexdigest()
    write(root / 'manifests/frozen_files.json', files)
    print(name, runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--releases', type=Path, required=True)
    args = parser.parse_args()
    for demos in (25, 50, 100, 200):
        build(args.releases, 'dp_c4_aug', demos, [0, 1, 2])
    for demos in (50, 200):
        build(args.releases, 'equidiff_c4', demos, [1, 2])
