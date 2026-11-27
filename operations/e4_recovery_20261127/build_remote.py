"""Create new continuation releases from read-only original manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p = argparse.ArgumentParser()
p.add_argument('--project', type=Path, required=True)
a = p.parse_args()
ops = Path(__file__).resolve().parent
project = a.project.resolve()
support = ops.parent / 'e4_recovery_20261126'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for seed, job in [(1, '29427'), (2, '29428')]:
    source = project / f'releases/e4_square_d0_s{seed}_sicc_20261123_v2'
    root = project / f'releases/e4_continue_s{seed}_20261127_v1'
    root.mkdir(exist_ok=False)
    for name, expected in json.loads((source / 'frozen_files.json').read_text()).items():
        assert sha(source / name) == expected, name
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, dest)
    for name in ['resume_support.py', 'evaluate.py', 'export.py', 'check_resume.py']:
        shutil.copy2(support / name, root / name)
    shutil.copy2(support / 'train.py', root / 'run.py')
    shutil.copy2(ops / 'continue_lane.py', root / 'lane.py')
    shutil.copy2(ops / 'scheduler_guard.py', root / 'scheduler_guard.py')
    for directory in ['runs', 'results', 'logs', 'preflight']:
        (root / directory).mkdir(exist_ok=True)
    shutil.copy2(source / 'results/data_ready.json', root / 'results/data_ready.json')
    plan = {'seed': seed, 'source_release': source.name, 'source_job': job,
            'source_manifest_sha256': sha(source / 'frozen_files.json'),
            'policy': 'Continue only after source terminates; preserve scientific YAML; evaluate independently'}
    (root / 'continuation.json').write_text(json.dumps(plan, indent=2)+'\n')
    (root / 'frozen_files.json').write_text(json.dumps({
        str(f.relative_to(root)): sha(f) for f in root.rglob('*')
        if f.is_file() and f.name != 'frozen_files.json'}, indent=2)+'\n')
    print(root)

# Register only new releases with the existing verified backup worker.
path = project / 'operations/parallel_20261119/registry.json'
registry = json.loads(path.read_text())
for seed in [1, 2]:
    name = f'e4_continue_s{seed}_20261127_v1'
    if name not in registry['releases']:
        registry['releases'].append(name)
tmp = path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(registry, indent=2)+'\n')
tmp.replace(path)
