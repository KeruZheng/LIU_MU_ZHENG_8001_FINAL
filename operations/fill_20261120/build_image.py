"""Freeze visible-goal image RQ3, reusing six existing rotated baselines."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main(base):
    source = base / 'image_rq3_20261120'
    primary = base / 'releases/c4_main_20261118_v1'
    # Shared baseline records must use exactly the primary data and test grid.
    for name in ('split.json', 'evaluation_seeds.json'):
        assert json.loads((source / 'manifests' / name).read_text()) == json.loads((primary / 'manifests' / name).read_text()), name
    shared = [f'c4_main_20261118_v1/{method}_n50_s{seed}'
              for method in ('dp', 'equidiff_c4') for seed in range(3)]
    for mode in ('fixed', 'rotated'):
        root = base / 'releases' / f'rq3_image_{mode}_20261120_v1'
        root.mkdir(exist_ok=False)
        for d in ('src', 'scripts', 'tests', 'manifests'):
            shutil.copytree(source / d, root / d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for d in ('configs', 'results', 'runs', 'logs'):
            (root / d).mkdir()
        for d in ('data', 'upstream'):
            (root / d).symlink_to(Path('../..') / d, target_is_directory=True)
        methods = ('dp', 'equidiff_c4', 'equidiff_c4_gate') if mode == 'fixed' else ('equidiff_c4_gate',)
        runs = []
        config = json.loads((primary / 'configs/equidiff_c4_n50_s0.json').read_text())
        for seed in range(3):
            for method in methods:
                run = f'{mode}_{method}_n50_s{seed}'
                runs.append(run)
                write(root / 'configs' / (run + '.json'), dict(config, method=method, seed=seed,
                    observation='image', goal_mode=mode, rotated=mode == 'rotated'))
        write(root / 'manifests/lane.json', {'runs': runs, 'milestones': [16000, 18000, 20000],
            'scope': 'Visible-goal image RQ3; post-state-result protocol', 'shared_rotated_baselines': shared})
        (root / 'PROTOCOL.md').write_text('''# Image RQ3: visible-goal extension

Frozen November 20, after inspecting state and primary image results. This is
an exploratory implementation of the proposal's image extension, not a new
pre-registered confirmation of the state-input hidden-goal hypothesis.

Matrix: fixed/rotated goal x {DP, EquiDiff-C4, EquiDiff-C4 + gate} x three seeds,
50 demos, 20k updates, batch64, [512,1024,2048] widths, unchanged EMA/optimizer,
100-step DDPM, FP32 with TF32 disabled. The fixed condition uses canonical demos;
the rotated condition uses the existing balanced C4 labels. Goal remains visible
in RGB for every method. An extra goal-input arm is not added because the image
already carries that information. A fixed-goal image does not have the state's
unobserved-goal ambiguity, so any deficit cannot be attributed to that mechanism
without further evidence.

The gate adds a fixed world vector (1,0) through a zero-initialized learnable
scalar into the equivariant observation projection. Zero gate preserves C4
equivariance; nonzero gate can break it. Unit tests verify both, gradient flow,
and visible goal differences with unchanged physics. Each allocation runs these
tests, full native replay, full-width batch64 backward, and short native rollout
checks before any formal training. Preflight failures prevent dispatch.

Evaluation: the existing 25 paired test base states x four initial orientations,
300 control steps, retain 16k/18k/20k checkpoints and report their mean. Fixed
condition keeps world goal fixed while initial states rotate. Coverage is
intersection/goal area; success is >0.95. State and image results stay separate.

There are 18 comparison cells but only 12 new trainings: the six completed
rotated DP/C4 50-demo primary runs are reused explicitly by global run ID,
not rerun or counted twice. Nine new fixed-goal runs and three rotated gate runs
use independent releases. This resolves the proposal's unspecified 18-image-run
layout without wasting six identical baseline trainings. The original proposal
described 18 additional runs; this is a documented deduplication of that budget.

Every complete run is copied locally and hash-verified before remote checkpoints
are deleted and the next run starts. No active primary source is modified.
''')
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for d in ('src', 'scripts', 'tests', 'manifests', 'configs') for p in (root / d).rglob('*') if p.is_file()}
        files['PROTOCOL.md'] = hashlib.sha256((root / 'PROTOCOL.md').read_bytes()).hexdigest()
        write(root / 'manifests/frozen_files.json', files)
        print(root.name, len(runs), 'new runs;', len(shared), 'shared baselines')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, type=Path)
    main(parser.parse_args().project)
