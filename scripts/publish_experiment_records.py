"""Export a dated, lightweight experiment notebook; never touch live releases."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_text(text, workspace):
    replacements = [
        (str(workspace), '${WORKSPACE}'),
        ('/data/qingbiaoli/yc67825/robotic', '${SICC_PROJECT}'),
        ('/home/user/yc67825/keru/robotic', '${LEGACY_PROJECT}'),
        ('/mnt/hdd1/keru', '${LOCAL_STORAGE}'),
        ('/home/zheng_keru', '${LOCAL_HOME}'),
        ('gpu.sicc.um.edu.mo', '${SICC_LOGIN}'),
        ('cluster-master.cis.um.edu.mo', '${LEGACY_LOGIN}'),
        ('yc67825', '${CLUSTER_USER}'),
        ('qingbiaoli', '${CLUSTER_ACCOUNT}'),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r'\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))(?:\.\d{1,3}){2,3}\b',
                  '${PRIVATE_IP}', text)


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for record in manifest['files']:
        path = root / record['published']
        assert sha(path.read_bytes()) == record['published_sha256'], path
    assert sha((root / 'release_inventory.json').read_bytes()) == manifest['release_inventory_sha256']
    print(f"Verified {len(manifest['files'])} published records and release inventory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--snapshot', required=True, help='New directory label, e.g. 2026-11-24')
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.snapshot)
    workspace = args.workspace.resolve()
    codes = Path(__file__).resolve().parents[1] if args.verify else workspace / 'codes'
    destination = codes / 'docs/experiment_records' / args.snapshot
    if args.verify:
        verify(destination)
        return
    assert not destination.exists(), 'Snapshots are immutable; choose a new label'
    sources = {}

    def include(source, target):
        source = workspace / source
        assert source.is_file() and not source.is_symlink(), source
        assert source.stat().st_size < 2 * 1024 * 1024, source
        sources[source.resolve()] = destination / target

    for path in sorted((workspace / 'notes').glob('*.md')):
        include(path.relative_to(workspace), 'notes/' + path.name)
    base = Path('development/proposal_v2')
    for name in ('PROTOCOL.md', 'RQ3_PROTOCOL.md', 'C8_PROTOCOL.md'):
        include(base / name, 'protocols/' + name)
    for path in sorted((workspace / base / 'releases').glob('*/PROTOCOL.md')):
        include(path.relative_to(workspace), 'protocols/releases/' + path.parent.name + '.md')
    for name in ('upstream.json', 'data.json', 'split.json', 'evaluation_seeds.json',
                 'experiment_matrix.json', 'verification.json'):
        include(base / 'manifests' / name, 'manifests/' + name)
    for name in ('c4_native_replay_v2.json', 'dataset_render_alignment_v2.json',
                 'equidiff_full_gpu_preflight.json', 'final_source_audit.json'):
        include(base / 'results' / name, 'evidence/' + name)
    include(base / 'results/e4_eval_audit_20261124/reset_audit.json', 'evidence/e4_reset_audit.json')
    for name in ('failed_v1_archive_verified.json', 'failed_v1_cleanup_receipt.json'):
        include(base / 'operations/e4_corrected_eval_20261124' / name, 'evidence/' + name)
    include('codes/tasks/rq1_transfer/results/e4_corrected_eval_20261124/v2/monitor.json',
            'evidence/e4_corrected_status_at_export.json')
    include(base / 'results/parallel_20261119/receipts/legacy/conv_c4_n25_s02_legacy_20261122_v1__equidiff_c4_n25_s2.json',
            'evidence/final_c4_archive_receipt.json')
    tracked = {codes / name for name in subprocess.check_output(
        ['git', '-C', str(codes), 'ls-files'], text=True).splitlines()}
    proposal = workspace / '03_Proposal_A_EquiDiff_final.md'
    published_proposal = codes / 'reports/main_results_20261123/proposal_snapshot.md'
    assert proposal.read_bytes() == published_proposal.read_bytes()
    links = dict(sources)
    links[proposal] = published_proposal
    captured = datetime.now(timezone.utc).isoformat()
    records = []
    for source, output in sources.items():
        raw = source.read_bytes()
        text = raw.decode('utf-8')
        unavailable = []
        if source.suffix == '.md':
            def rewrite(match):
                label, href = match.groups()
                if re.match(r'(?:https?://|mailto:|#)', href):
                    return match.group(0)
                name, sep, fragment = href.partition('#')
                resolved = (source.parent / name).resolve()
                target = links.get(resolved, resolved if resolved in tracked else None)
                if target is None:
                    unavailable.append(href)
                    return f'{label} (`{href}`; local-only historical reference)'
                link = os.path.relpath(target, output.parent) + (sep + fragment if sep else '')
                return f'[{label}]({link})'
            text = re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', rewrite, text)
            text = (f'> Historical record exported {captured}. Source: '
                    f'`{source.relative_to(workspace)}`. Statuses and instructions below '
                    'describe their original dates, not the current queue. '
                    'Private connection values are placeholders; local-only references '
                    'are labelled. See the experiment-record index for current interpretation.\n\n' + text)
        text = public_text(text, workspace)
        if source.suffix == '.json':
            json.loads(text)
        assert not re.search(r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]+|-----BEGIN .*PRIVATE KEY', text)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
        records.append({'source': str(source.relative_to(workspace)),
                        'source_sha256': sha(raw), 'source_bytes': len(raw),
                        'published': str(output.relative_to(destination)),
                        'published_sha256': sha(output.read_bytes()),
                        'published_bytes': output.stat().st_size,
                        'local_only_references': unavailable})
    inventory = []
    for release in sorted((workspace / base / 'releases').iterdir()):
        if not release.is_dir():
            continue
        public = codes / 'experiments/releases' / release.name / 'SNAPSHOT.json'
        frozen = release / 'frozen_files.json'
        inventory.append({'release': release.name,
            'local_path': str(release.relative_to(workspace)),
            'published_scientific_snapshot': str(public.relative_to(codes)) if public in tracked else None,
            'frozen_manifest_sha256': sha(frozen.read_bytes()) if frozen.exists() else None,
            'protocol': 'protocols/releases/' + release.name + '.md'
                        if (release / 'PROTOCOL.md').exists() else None})
    inventory_path = destination / 'release_inventory.json'
    inventory_path.write_text(json.dumps({'captured_at': captured,
        'scope': 'Directory inventory, not training completion or full-source verification',
        'releases': inventory}, indent=2) + '\n')
    manifest = {'captured_at': captured, 'source_root': 'wangdian/', 'snapshot': args.snapshot,
        'transformations': ['private connection/path placeholders', 'markdown link remapping',
                            'historical-record banners'],
        'excluded': ['environments', 'datasets', 'checkpoints', 'bulk logs', 'private cluster configs',
                     'historical executable scripts in notes (documentary references only)'],
        'release_inventory_sha256': sha(inventory_path.read_bytes()), 'files': records}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    lines = ['# Dated Experiment Records', '', f'Captured (UTC): {captured}', '',
             'These are historical snapshots. Read [the main index](../README.md) first.', '',
             '| Published record | Original workspace source |', '|---|---|']
    lines += [f"| [{r['published']}]({r['published']}) | `{r['source']}` |" for r in records]
    lines += ['', '[Source and export hashes](manifest.json) / [Release inventory](release_inventory.json)', '']
    (destination / 'README.md').write_text('\n'.join(lines))
    verify(destination)


if __name__ == '__main__':
    main()
