"""Check manually typeset tables against the frozen numerical summary."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SUMMARY = json.loads((ROOT.parent / 'summary.json').read_text())
METHODS = ('dp', 'dp_c4_aug', 'equidiff_c4')


def group(method, demos=50, observation='image', goal='rotated', split='test'):
    return next(r for r in SUMMARY['table'] if r['method'] == method
                and r['demos'] == demos and r['observation'] == observation
                and r['goal_mode'] == goal and r['split'] == split)


def difference(left, right, demos=50):
    for row in SUMMARY['paired_comparisons']:
        if row['observation'] != 'image' or row['goal_mode'] != 'rotated' or row['demos'] != demos:
            continue
        d = row['coverage']
        if (row['left'], row['right']) == (left, right):
            return [d['mean_difference'], *d['ci95']]
        if (row['left'], row['right']) == (right, left):
            return [-d['mean_difference'], -d['ci95'][1], -d['ci95'][0]]
    raise AssertionError((left, right, demos))


def rows(tex, label):
    tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', tex, re.S)
    table = next(t for t in tables if '\\label{' + label + '}' in t)
    body = table.split('\\midrule', 1)[1].split('\\bottomrule', 1)[0]
    return [line for line in body.splitlines() if '&' in line]


def check_values(line, expected, digits=4):
    actual = re.findall(r'[-+]?\d+\.\d+', line)
    assert len(actual) == len(expected), (line, expected)
    assert [f'{float(x):.{digits}f}' for x in actual] == [f'{x:.{digits}f}' for x in expected], (line, expected)


checks = 0
for language in ('en', 'zh'):
    tex = (ROOT / f'body_{language}.tex').read_text()
    prefix = 'eqstudy:tab:' if language == 'en' else 'eqstudy:zh:tab:'
    for n, line in zip((25, 50, 100, 200), rows(tex, prefix + 'main'), strict=True):
        expected = []
        for method in METHODS:
            r = group(method, n)
            expected += [r['mean_coverage'], r['seed_std']]
        check_values(line, expected)
        checks += 1
    conv_label = 'convergence' if language == 'en' else 'conv'
    for n, line in zip((25, 50, 100, 200), rows(tex, prefix + conv_label), strict=True):
        expected = []
        for method in METHODS:
            r = group(method, n, split='validation')
            expected += [r['per_checkpoint_mean'][0], r['per_checkpoint_mean'][-1]]
        check_values(line, expected)
        checks += 1
    for goal, line in zip(('fixed', 'rotated'), rows(tex, prefix + 'state'), strict=True):
        expected = []
        for method in ('state_dp', 'state_equidiff', 'state_equidiff_goal', 'state_equidiff_gate'):
            r = group(method, observation='state', goal=goal)
            expected += [r['mean_coverage'], r['seed_std']]
        check_values(line, expected)
        checks += 1
    for n, line in zip((25, 50, 100, 200), rows(tex, prefix + 'success'), strict=True):
        check_values(line, [100 * group(m, n)['success_rate'] for m in METHODS], digits=2)
        checks += 1
    if language == 'en':
        for n, line in zip((25, 50, 100, 200), rows(tex, prefix + 'deltas'), strict=True):
            check_values(line, difference('equidiff_c4', 'dp', n) + difference('equidiff_c4', 'dp_c4_aug', n))
            checks += 1
    else:
        for table, right in (('rq1', 'dp'), ('rq2', 'dp_c4_aug')):
            for n, line in zip((25, 50, 100, 200), rows(tex, prefix + table), strict=True):
                check_values(line, difference('equidiff_c4', right, n))
                checks += 1
        for goal, line in zip(('fixed', 'rotated'), rows(tex, prefix + 'image'), strict=True):
            check_values(line, [group(m, goal=goal)['mean_coverage'] for m in ('dp', 'equidiff_c4', 'equidiff_c4_gate')])
            checks += 1
        pairs = [('state_equidiff', 'state_dp'), ('state_equidiff_goal', 'state_equidiff'),
                 ('state_equidiff_gate', 'state_equidiff'), ('state_equidiff_goal', 'state_dp'),
                 ('state_equidiff_gate', 'state_dp'), ('state_equidiff_gate', 'state_equidiff_goal')]
        for (left, right), line in zip(pairs, rows(tex, prefix + 'repair'), strict=True):
            r = next(r for r in SUMMARY['paired_comparisons'] if r['observation'] == 'state'
                     and r['goal_mode'] == 'fixed' and {r['left'], r['right']} == {left, right})
            d = r['coverage']
            expected = [d['mean_difference'], *d['ci95']]
            if r['left'] != left:
                expected = [-expected[0], -expected[2], -expected[1]]
            check_values(line, expected)
            checks += 1
    citations = {key.strip() for match in re.findall(r'\\cite\{([^}]+)\}', tex) for key in match.split(',')}
    bib = set(re.findall(r'@\w+\{([^,]+),', (ROOT / 'references.bib').read_text()))
    assert citations <= bib
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    assert len(labels) == len(set(labels))
    assert set(re.findall(r'\\ref\{([^}]+)\}', tex)) <= set(labels)
    for figure in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex):
        assert (ROOT / 'figures' / figure).is_file(), figure

print(json.dumps(dict(table_rows_verified=checks, citations_and_labels='passed',
    figure_paths='passed', summary_sha256=hashlib.sha256((ROOT.parent / 'summary.json').read_bytes()).hexdigest())))
