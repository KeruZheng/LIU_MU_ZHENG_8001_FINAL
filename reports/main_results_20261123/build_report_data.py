"""Freeze completed task evidence and rebuild report tables/figures on CPU."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODES = HERE.parents[1]
TASKS = ('rq1_transfer', 'rq2_exact_aug', 'rq3_symmetry_breaking')


def read(path):
    return json.loads(path.read_text())


def save(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + '\n')


def capture():
    entries, manifests = {}, {}
    for task in TASKS:
        path = CODES / 'tasks' / task / 'reports/latest.json'
        payload = path.read_bytes()
        manifests[str(path.relative_to(CODES))] = hashlib.sha256(payload).hexdigest()
        for run in json.loads(payload)['runs']:
            if run['run_id'] in entries:
                assert entries[run['run_id']]['evidence'] == run['evidence']
                continue
            assert run['state'] == 'complete', run['run_id']
            entries[run['run_id']] = run
    records = []
    for run_id, entry in sorted(entries.items()):
        root = CODES / entry['evidence']
        config_path = root / 'config.json'
        config = read(config_path)
        manifests[str(config_path.relative_to(CODES))] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        split = entry['split']
        steps = [16000, 18000, 20000] if split == 'test' else [20000, 40000, 60000]
        record = {k: entry[k] for k in ('run_id', 'study', 'split', 'evidence', 'code_snapshot', 'snapshot_sha256', 'runtime_gpu', 'torch')}
        record.update(method=config['method'], demos=config['demos'],
                      seed=config['seed'], observation=config.get('observation', 'image'),
                      goal_mode=config.get('goal_mode', 'rotated'), checkpoints={})
        for step in steps:
            path = root / f'{split}_step_{step}.json'
            payload = path.read_bytes()
            result = json.loads(payload)
            assert result['state'] == 'complete'
            assert result['checkpoint_step'] == step
            assert result['signature']['split'] == split
            assert result['training_seed'] == record['seed']
            assert result['method'] == record['method']
            rows = sorted(result['episode_results'], key=lambda x: (x['base_seed'], x['orientation']))
            keys = [(x['base_seed'], x['orientation']) for x in rows]
            bases = sorted({x[0] for x in keys})
            assert len(bases) == 25 and keys == [(b, k) for b in bases for k in range(4)]
            if 'episode_keys' in record:
                assert record['episode_keys'] == keys
            record['episode_keys'] = keys
            coverage = [x['max_coverage'] for x in rows]
            success = [float(x['success']) for x in rows]
            assert np.isclose(np.mean(coverage), result['summary']['mean_max_coverage'], atol=1e-12)
            assert np.isclose(np.mean(coverage), entry['scores'][str(step)]['mean_max_coverage'], atol=1e-12)
            assert all(0 <= x <= 1 for x in coverage)
            record['checkpoints'][str(step)] = dict(coverage=coverage, success=success,
                checkpoint_sha256=result['signature']['checkpoint_sha256'])
            manifests[str(path.relative_to(CODES))] = hashlib.sha256(payload).hexdigest()
        records.append(record)
    assert len(records) == 114
    assert sum(x['split'] == 'test' for x in records) == 78
    assert sum(x['split'] == 'validation' for x in records) == 36
    save('evidence.json', dict(captured_at=datetime.now(timezone.utc).isoformat(),
        episode_order='base_seed then orientation; checkpoints are repeated measures, not independent runs',
        runs=records))
    save('source_hashes.json', manifests)


def bootstrap(left, right):
    # Resample the crossed seed/state factors; keep the four orientations together.
    delta = (left - right).mean(axis=2)
    rng = np.random.default_rng(20261118)
    train = rng.integers(delta.shape[0], size=(10000, delta.shape[0]))
    bases = rng.integers(delta.shape[1], size=(10000, delta.shape[1]))
    values = delta[train[:, :, None], bases[:, None, :]].mean(axis=(1, 2))
    return dict(mean_difference=float(delta.mean()), ci95=np.quantile(values, [.025, .975]).tolist())


def analyze():
    evidence = read(HERE / 'evidence.json')
    groups = defaultdict(list)
    for r in evidence['runs']:
        groups[(r['split'], r['observation'], r['goal_mode'], r['demos'], r['method'])].append(r)
    table, arrays = [], {}
    for key, runs in sorted(groups.items()):
        runs.sort(key=lambda r: r['seed'])
        assert [r['seed'] for r in runs] == [0, 1, 2]
        assert all(r['episode_keys'] == runs[0]['episode_keys'] for r in runs)
        steps = [16000, 18000, 20000] if key[0] == 'test' else [20000, 40000, 60000]
        cov = np.array([[r['checkpoints'][str(s)]['coverage'] for s in steps] for r in runs])
        suc = np.array([[r['checkpoints'][str(s)]['success'] for s in steps] for r in runs])
        row = dict(zip(('split', 'observation', 'goal_mode', 'demos', 'method'), key))
        row.update(steps=steps, per_checkpoint_mean=cov.mean(axis=(0, 2)).tolist(),
                   per_seed_checkpoint_mean=cov.mean(axis=2).tolist(),
                   per_checkpoint_success=suc.mean(axis=(0, 2)).tolist(), run_ids=[r['run_id'] for r in runs])
        if key[0] == 'test':
            values = cov.mean(axis=1).reshape(3, 25, 4)
            successes = suc.mean(axis=1).reshape(3, 25, 4)
            per_seed = values.mean(axis=(1, 2))
            row.update(mean_coverage=float(values.mean()), seed_std=float(per_seed.std(ddof=1)),
                       seed_coverages=per_seed.tolist(), success_rate=float(successes.mean()),
                       coverage_ci95=bootstrap(values, np.zeros_like(values))['ci95'],
                       limited_grid_max_mean=float(cov.mean(axis=2).max(axis=1).mean()),
                       per_orientation=values.mean(axis=(0, 1)).tolist())
            arrays[key] = (values, successes, runs[0]['episode_keys'])
        table.append(row)
    comparisons = []
    for left, right in combinations(sorted(arrays), 2):
        if left[:-1] != right[:-1]:
            continue
        lv, ls, lk = arrays[left]
        rv, rs, rk = arrays[right]
        assert lk == rk
        comparisons.append(dict(observation=left[1], goal_mode=left[2], demos=left[3],
            left=left[4], right=right[4], coverage=bootstrap(lv, rv), success=bootstrap(ls, rs)))
    save('summary.json', dict(captured_at=evidence['captured_at'], unique_runs=len(evidence['runs']),
        test_runs=78, validation_runs=36, bootstrap_repetitions=10000, bootstrap_seed=20261118,
        multiplicity_correction=False, table=table, paired_comparisons=comparisons))
    lines = ['# Computed tables', '', 'Coverage is intersection / goal area, not success or IoU.', '',
             '## Test: mean over 16k / 18k / 20k', '',
             '| Input | Goal | Demos | Method | Coverage | Seed SD | Success | 20k only | Limited grid max |',
             '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in table:
        if r['split'] == 'test':
            lines.append(f"| {r['observation']} | {r['goal_mode']} | {r['demos']} | {r['method']} | {r['mean_coverage']:.4f} | {r['seed_std']:.4f} | {100*r['success_rate']:.2f}% | {r['per_checkpoint_mean'][-1]:.4f} | {r['limited_grid_max_mean']:.4f} |")
    lines += ['', '## Independent validation curves', '', '| Demos | Method | 20k | 40k | 60k |', '|---:|---|---:|---:|---:|']
    for r in table:
        if r['split'] == 'validation':
            lines.append(f"| {r['demos']} | {r['method']} | " + ' | '.join(f'{v:.4f}' for v in r['per_checkpoint_mean']) + ' |')
    lines += ['', '## Paired coverage differences (left minus right)', '',
              '| Input | Goal | Demos | Left | Right | Difference | 95% CI |', '|---|---|---:|---|---|---:|---|']
    for r in comparisons:
        c = r['coverage']
        lines.append(f"| {r['observation']} | {r['goal_mode']} | {r['demos']} | {r['left']} | {r['right']} | {c['mean_difference']:.4f} | [{c['ci95'][0]:.4f}, {c['ci95'][1]:.4f}] |")
    (HERE / 'tables.md').write_text('\n'.join(lines) + '\n')
    figures(table)
    print(json.dumps(dict(unique_runs=len(evidence['runs']), groups=len(table), comparisons=len(comparisons))))


def figures(table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none'})
    colors = {'dp': '#555555', 'dp_c4_aug': '#007f78', 'equidiff_c4': '#c36326'}
    labels = {'dp': 'DP', 'dp_c4_aug': 'DP + exact C4 aug.', 'equidiff_c4': 'EquiDiff-C4'}
    def finish(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(HERE / f'{name}.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for method in colors:
        rows = [r for r in table if r['split']=='test' and r['observation']=='image' and r['goal_mode']=='rotated' and r['method']==method]
        ax.plot([r['demos'] for r in rows], [r['mean_coverage'] for r in rows], marker='o', color=colors[method], label=labels[method])
        for r in rows:
            ax.scatter(np.array([-2, 0, 2])+r['demos'], r['seed_coverages'], s=20, alpha=.5, color=colors[method])
    ax.set(xlabel='Independent training demonstrations', ylabel='Mean episode-max coverage', ylim=(0, 1), xticks=[25,50,100,200],
           title='RQ1 / RQ2: reserved test, 20k-update budget\nLines: three-seed mean; small dots: individual seeds')
    ax.legend(); ax.grid(alpha=.2); finish(fig, 'rq1_rq2_test')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    methods = ['state_dp', 'state_equidiff', 'state_equidiff_goal', 'state_equidiff_gate']
    for ax, goal in zip(axes, ['fixed', 'rotated']):
        rows = [next(r for r in table if r['split']=='test' and r['goal_mode']==goal and r['method']==m) for m in methods]
        ax.bar(range(4), [r['mean_coverage'] for r in rows], color=['#555555','#c36326','#007f78','#326b99'])
        for i,r in enumerate(rows):
            ax.scatter(np.array([-.09,0,.09])+i, r['seed_coverages'], color='black', s=15)
        ax.set(xticks=range(4), xticklabels=['DP','EquiDiff','+ goal','+ gate'], ylim=(0,1),title=f'{goal.title()} goal, state inputs')
    axes[0].set_ylabel('Mean episode-max coverage')
    fig.suptitle('RQ3: reserved test; dots are training seeds')
    finish(fig, 'rq3_state_test')
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7), sharey=True)
    for ax,n in zip(axes,[25,50,100,200]):
        for method in colors:
            r=next(r for r in table if r['split']=='validation' and r['demos']==n and r['method']==method)
            ax.plot([20,40,60], r['per_checkpoint_mean'], marker='o', color=colors[method], label=labels[method])
        ax.set(title=f'{n} demos', xlabel='Updates (thousands)', ylim=(0,1), xticks=[20,40,60]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Validation mean episode-max coverage'); axes[-1].legend(fontsize=8)
    fig.suptitle('Independent convergence checks: validation only, three seeds per curve')
    finish(fig, 'convergence_validation')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', action='store_true', help='Read completed local task evidence before rebuilding')
    args = parser.parse_args()
    if args.capture:
        capture()
    analyze()
