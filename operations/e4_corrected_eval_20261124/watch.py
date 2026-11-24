"""Pull corrected evaluation evidence; verify before removing remote EMA duplicate."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from evaluate import atomic, digest

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True)
parser.add_argument('--host-alias', required=True)
parser.add_argument('--remote', required=True)
parser.add_argument('--local', type=Path, required=True)
parser.add_argument('--weights-backup', type=Path, required=True)
parser.add_argument('--job', required=True)
args = parser.parse_args()
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'HostKeyAlias=' + args.host_alias, args.host]
args.local.mkdir(parents=True, exist_ok=True)


def remote(command):
    return subprocess.check_output(ssh + [command], text=True, timeout=60)


while True:
    try:
        subprocess.run(['rsync', '-a', '--partial', '-e', shlex.join(ssh[:-1]),
            args.host + ':' + args.remote + '/results/', str(args.local / 'results') + '/'],
            check=True, timeout=120)
        queue = remote('squeue -h -j ' + shlex.quote(args.job) + ' -o ' + shlex.quote('%i %T %M %N'))
        path = args.local / 'results/equidiff_s0/status.json'
        state = json.loads(path.read_text()) if path.exists() else {'state': 'waiting'}
        atomic(args.local / 'monitor.json', {'time': time.time(), 'queue': queue.strip(), 'evaluation': state})
        (args.local / 'STATUS.md').write_text(
            '# Corrected E4 evaluation\n\n'
            f'Job: {args.job}\n\nQueue: {queue.strip() or "not active"}\n\n'
            f'State: {state["state"]}; completed episodes: {state.get("episodes_completed", 0)}/300.\n\n'
            'Old D1/D2 outputs are provisional because of the scene reset defect.\n'
            'Main RQ1/RQ2/RQ3 results are unaffected.\n')
        if state['state'] == 'failed':
            break
        if not queue.strip() and state['state'] != 'complete':
            accounting = remote('sacct -n -X -j ' + shlex.quote(args.job) + ' -o State,ExitCode')
            if any(word in accounting for word in ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'COMPLETED')):
                atomic(args.local / 'terminal_without_results.json', {
                    'time': time.time(), 'accounting': accounting, 'evaluation': state})
                break
        if state['state'] == 'complete' and not queue.strip():
            # Hash all completed evidence, including per-episode scene records.
            code = ('from pathlib import Path; import hashlib,json; '
                    'p=Path(' + repr(args.remote + '/results') + '); '
                    'print(json.dumps({str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() '
                    'for f in p.rglob("*") if f.is_file()}))')
            hashes = json.loads(remote('python3 -c ' + shlex.quote(code)))
            assert hashes and all(digest(args.local / 'results' / name) == value
                                  for name, value in hashes.items())
            manifest_path = args.weights_backup.parent / 'manifest.json'
            manifest = json.loads(manifest_path.read_text())
            expected = manifest['weights_sha256']
            assert digest(args.weights_backup) == expected
            weights = args.remote + '/inputs/equidiff_s0/final_ema.pt'
            assert remote('sha256sum ' + shlex.quote(weights)).split()[0] == expected
            atomic(args.local / 'archive_verified.json', {'state': 'verified', 'time': time.time(),
                'result_hashes': hashes, 'weights_sha256': expected, 'backup': str(args.weights_backup),
                'job': args.job})
            remote('rm -- ' + shlex.quote(weights))
            atomic(args.local / 'cleanup_receipt.json', {'state': 'remote_ema_duplicate_removed',
                'time': time.time(), 'path': weights, 'sha256': expected})
            break
    except Exception as exc:
        atomic(args.local / 'monitor_error.json', {'time': time.time(), 'error': repr(exc)})
    time.sleep(60)
