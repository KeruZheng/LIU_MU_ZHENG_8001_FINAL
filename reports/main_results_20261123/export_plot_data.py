"""Export plotting CSVs from frozen report evidence; no GPU or live-data reads."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from math import isclose
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
DIMENSIONS = ['split', 'observation', 'goal_mode', 'demos', 'method']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'plot_data')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((HERE / 'evidence.json').read_text())
    summary = json.loads((HERE / 'summary.json').read_text())
    before = {name: digest(HERE / name) for name in ('evidence.json', 'summary.json', 'source_hashes.json')}
    files = {}

    def write(name, rows):
        assert rows
        path = args.output / name
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        with path.open(newline='') as stream:
            assert len(list(csv.DictReader(stream))) == len(rows)
        files[name] = {'rows': len(rows), 'sha256': digest(path)}

    runs = {r['run_id']: r for r in evidence['runs']}
    episodes, run_checkpoints, run_catalog = [], [], []
    for run in evidence['runs']:
        dimensions = {key: run[key] for key in DIMENSIONS}
        run_catalog.append({**dimensions, **{key: run[key] for key in
            ('run_id', 'seed', 'study', 'evidence', 'code_snapshot', 'snapshot_sha256', 'runtime_gpu', 'torch')}})
        for step, checkpoint in run['checkpoints'].items():
            assert len(checkpoint['coverage']) == len(checkpoint['success']) == len(run['episode_keys']) == 100
            run_checkpoints.append({**dimensions, 'run_id': run['run_id'], 'training_seed': run['seed'],
                'checkpoint_step': int(step), 'mean_coverage': mean(checkpoint['coverage']),
                'success_rate': mean(checkpoint['success']), 'episodes': 100,
                'checkpoint_sha256': checkpoint['checkpoint_sha256']})
            for (base_seed, orientation), coverage, success in zip(
                    run['episode_keys'], checkpoint['coverage'], checkpoint['success']):
                episodes.append({**dimensions, 'run_id': run['run_id'], 'training_seed': run['seed'],
                    'checkpoint_step': int(step), 'base_seed': base_seed, 'orientation': orientation,
                    'max_coverage': coverage, 'success': int(success)})
    assert len(run_catalog) == 114 and len(run_checkpoints) == 342 and len(episodes) == 34200
    write('run_catalog.csv', run_catalog)
    write('run_checkpoints.csv', run_checkpoints)
    write('episodes.csv', episodes)

    tests, seeds, checkpoints = [], [], []
    for row in summary['table']:
        dimensions = {key: row[key] for key in DIMENSIONS}
        group = [runs[rid] for rid in row['run_ids']]
        assert [r['seed'] for r in group] == [0, 1, 2]
        for i, step in enumerate(row['steps']):
            values = [mean(r['checkpoints'][str(step)]['coverage']) for r in group]
            successes = [mean(r['checkpoints'][str(step)]['success']) for r in group]
            assert isclose(mean(values), row['per_checkpoint_mean'][i], abs_tol=1e-12)
            assert isclose(mean(successes), row['per_checkpoint_success'][i], abs_tol=1e-12)
            checkpoints.append({**dimensions, 'checkpoint_step': step,
                'mean_coverage': row['per_checkpoint_mean'][i], 'seed_std': stdev(values),
                'success_rate': row['per_checkpoint_success'][i],
                **{f'seed_{seed}_coverage': values[seed] for seed in range(3)}})
        if row['split'] != 'test':
            continue
        values = [mean(mean(r['checkpoints'][str(step)]['coverage']) for step in row['steps']) for r in group]
        assert isclose(mean(values), row['mean_coverage'], abs_tol=1e-12)
        assert isclose(stdev(values), row['seed_std'], abs_tol=1e-12)
        tests.append({**dimensions, 'aggregation': 'mean_16k_18k_20k',
            **{key: row[key] for key in ('mean_coverage', 'seed_std', 'success_rate', 'limited_grid_max_mean')},
            'coverage_ci95_low': row['coverage_ci95'][0], 'coverage_ci95_high': row['coverage_ci95'][1],
            'final_20k_coverage': row['per_checkpoint_mean'][-1],
            **{f'seed_{seed}_coverage': row['seed_coverages'][seed] for seed in range(3)}})
        for seed, r in enumerate(group):
            assert isclose(values[seed], row['seed_coverages'][seed], abs_tol=1e-12)
            seeds.append({**dimensions, 'run_id': r['run_id'], 'training_seed': seed,
                'aggregation': 'mean_16k_18k_20k', 'mean_coverage': values[seed],
                'success_rate': mean(mean(r['checkpoints'][str(step)]['success']) for step in row['steps'])})
    write('test_summary.csv', tests)
    write('test_seeds.csv', seeds)
    write('checkpoint_summary.csv', checkpoints)
    write('rq1_rq2.csv', [r for r in tests if r['observation'] == 'image' and r['goal_mode'] == 'rotated'
                         and r['method'] in ('dp', 'dp_c4_aug', 'equidiff_c4')])
    write('rq3_state.csv', [r for r in tests if r['observation'] == 'state'])
    write('rq3_image.csv', [r for r in tests if r['observation'] == 'image' and r['demos'] == 50
                          and r['method'] in ('dp', 'equidiff_c4', 'equidiff_c4_gate')])
    write('c8.csv', [r for r in tests if r['method'] == 'equidiff_c8'])
    write('convergence.csv', [r for r in checkpoints if r['split'] == 'validation'])
    comparisons = []
    for row in summary['paired_comparisons']:
        for metric in ('coverage', 'success'):
            result = row[metric]
            comparisons.append({**{key: row[key] for key in ('observation', 'goal_mode', 'demos', 'left', 'right')},
                'split': 'test', 'aggregation': 'mean_16k_18k_20k', 'metric': metric,
                'difference_left_minus_right': result['mean_difference'],
                'ci95_low': result['ci95'][0], 'ci95_high': result['ci95'][1]})
    write('paired_comparisons.csv', comparisons)
    # Recompute from serialized episode CSV, checking the public flat format itself.
    groups = defaultdict(list)
    with (args.output / 'episodes.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            groups[(row['run_id'], int(row['checkpoint_step']))].append(float(row['max_coverage']))
    for row in run_checkpoints:
        assert isclose(mean(groups[row['run_id'], row['checkpoint_step']]), row['mean_coverage'], abs_tol=1e-12)
    assert before == {name: digest(HERE / name) for name in before}
    provenance = {'frozen_evidence_captured_at': evidence['captured_at'], 'input_sha256': before,
        'export_script_sha256': digest(Path(__file__)), 'files': files,
        'verification': {'run_checkpoint_means_recomputed_from_export': len(groups),
                         'group_means_and_seed_std_checked': len(summary['table']),
                         'input_files_unchanged': True},
        'interpretation': 'Checkpoint episodes are repeated measures, not independent training runs. '
                          'Coverage is intersection/goal area. Success rates are fractions. E4 excluded.'}
    (args.output / 'manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({name: record['rows'] for name, record in files.items()}, indent=2))


if __name__ == '__main__':
    main()
