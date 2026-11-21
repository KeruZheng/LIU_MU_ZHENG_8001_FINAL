"""Publish byte-identical scientific snapshots without moving active releases."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

CODES = Path(__file__).resolve().parents[1]
NAMES = ['c4_main_20261118_v1', 'rq3_state_20261118_v1', 'c8_ablation_20261118_v1']
NAMES += ['conv_dp_n%s_20261119_v1' % n for n in (25, 50, 100, 200)]
NAMES += ['conv_c4_n%s_20261119_v2' % n for n in (50, 200)]
NAMES += ['c8_speed_20261119_v1', 'official_dp_preflight_20261119_v1']
NAMES += ['official_dp_s42_20261120_v1']
NAMES += [f'conv_aug_n{n}_20261120_v1' for n in (25, 50, 100, 200)]
NAMES += [f'conv_c4_n{n}_s12_20261120_v1' for n in (50, 200)]
NAMES += [f'rq3_image_{mode}_20261120_v1' for mode in ('fixed', 'rotated')]
NAMES += ['conv_c4_n25_n100_local_20261122_v1']
MANIFESTS = {'data.json', 'data.sha256', 'split.json', 'evaluation_seeds.json',
             'upstream.json', 'equidiff_commit.json', 'diffusion_policy_commit.json',
             'lane.json'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package(source, destination):
    original = json.loads((source / 'manifests/frozen_files.json').read_text())
    selected = {}
    for relative, expected in original.items():
        parts = Path(relative).parts
        keep = parts[0] in ('src', 'configs', 'tests')
        keep |= len(parts) == 1 and relative.endswith('.md')
        keep |= relative in ('config.yaml', 'upstream_files.json', 'runner.py', 'train.sbatch')
        keep |= parts[0] == 'manifests' and Path(relative).name in MANIFESTS
        keep |= relative in ('scripts/validate_c4.py', 'scripts/image_preflight.py', 'scripts/image_lane.py')
        keep |= source.name == 'conv_c4_n25_n100_local_20261122_v1' and relative == 'scripts/supervise.py'
        if not keep:
            continue
        path = source / relative
        if digest(path) != expected:
            raise ValueError('Source no longer matches frozen manifest: ' + str(path))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != expected:
            raise FileExistsError('Refusing to overwrite changed snapshot: ' + str(target))
        if not target.exists():
            shutil.copy2(path, target)
        selected[relative] = expected
    # Some convergence manifests were frozen before lane.json was included.
    for name in MANIFESTS:
        path = source / 'manifests' / name
        if path.is_file() and 'manifests/' + name not in selected:
            target = destination / 'manifests' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != digest(path):
                raise FileExistsError(str(target))
            shutil.copy2(path, target)
            selected['manifests/' + name] = digest(path)
    record = {'release': source.name,
              'original_frozen_manifest_sha256': digest(source / 'manifests/frozen_files.json'),
              'files': selected,
              'scope': 'Exact scientific files; machine-specific queues, environments and logs omitted'}
    (destination / 'SNAPSHOT.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    for name in NAMES:
        record = package(args.source / name, CODES / 'experiments/releases' / name)
        print(name, len(record['files']), 'verified files')
