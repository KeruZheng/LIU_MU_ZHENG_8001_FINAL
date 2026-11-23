"""Freeze SICC seed lanes from verified local E4 source, changing GPU admission only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    source = project / 'releases/e4_square_d0_local_20261122_v1'
    ops = Path(__file__).resolve().parent
    frozen = json.loads((source / 'frozen_files.json').read_text())
    for name, expected in frozen.items():
        assert sha(source / name) == expected, name
    runtime = json.loads((source / 'preflight/equidiff_square_d0_s0/runtime.json').read_text())
    for seed in (1, 2):
        root = project / f'releases/e4_square_d0_s{seed}_sicc_20261123_v1'
        root.mkdir(exist_ok=False)
        for name in frozen:
            if name == 'supervise.py':
                continue
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        runner = (root / 'run.py').read_text()
        old = "assert torch.cuda.is_available() and '5090' in torch.cuda.get_device_name()"
        assert runner.count(old) == 1
        runner = runner.replace(old, "assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8")
        (root / 'run.py').write_text(runner)
        for name in ('lane.py', 'train.sbatch'):
            shutil.copy2(ops / name, root / name)
        shutil.copy2(project / 'operations/parallel_20261119/transfer_guard.py', root / 'transfer_guard.py')
        for directory in ('results', 'logs', 'runs', 'preflight'):
            (root / directory).mkdir(exist_ok=True)
        shutil.copy2(source / 'results/data_ready.json', root / 'results/data_ready.json')
        keys = ('target_updates', 'batch_size', 'train_episode_ids', 'validation_episode_ids', 'training_input_sha256')
        lane = {'seed': seed, 'source_release': source.name, 'source_manifest_sha256': sha(source / 'frozen_files.json'),
            'methods': ['equidiff', 'dp'], 'reference_runtime': {k: runtime[k] for k in keys},
            'adaptations': ['GPU admission: CUDA compute capability >= 8 instead of RTX 5090 name',
                'PyTorch 2.7.1 CUDA 12.8 project environment instead of local 2.12 nightly',
                'Seed ownership and verified local archive gate; scientific configurations unchanged']}
        (root / 'lane.json').write_text(json.dumps(lane, indent=2) + '\n')
        with (root / 'PROTOCOL.md').open('a') as stream:
            stream.write('\n## SICC placement, November 23\n\n'
                f'This immutable lane owns seed {seed}, EquiDiff followed by DP. '
                'Local seed 0 stays on the RTX 5090. Both models pass full-size preflight on the allocated '
                'GPU before formal training. Scientific YAML files are byte-identical to the local release. '
                'CUDA admission supports A100/H800. Runtime versions and GPU are recorded. '
                'Each complete run is copied locally and hash-verified before remote checkpoint cleanup '
                'and dispatch of the next method. This replaces local placement for seeds 1 and 2.\n')
        files = {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}
        (root / 'frozen_files.json').write_text(json.dumps(files, indent=2) + '\n')
        print(root)


if __name__ == '__main__':
    main()
