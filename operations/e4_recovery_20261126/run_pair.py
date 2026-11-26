"""Run both seed-zero policies under the same independently checked protocol."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluate import atomic, digest

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--upstream', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
with (a.output / 'pair.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {'pid': os.getpid(), 'state': 'starting', 'started_at': time.time(),
              'protocol': 'seeded_fresh_scene_v3', 'completed_models': 0}
    try:
        for name, expected in json.loads((a.root / 'frozen_files.json').read_text()).items():
            assert digest(a.root / name) == expected, name
        for method in ['dp', 'equidiff']:
            inputs = a.root / 'inputs' / f'{method}_s0'
            status.update(state='evaluating', method=method, heartbeat=time.time())
            atomic(a.output / 'pair_status.json', status)
            command = [sys.executable, '-u', str(a.root / 'evaluate.py'), 'suite',
                       '--upstream', str(a.upstream), '--config', str(inputs / 'config.yaml'),
                       '--weights', str(inputs / 'final_ema.pt'), '--metadata-dir', str(inputs / 'metadata'),
                       '--output', str(a.output / f'{method}_s0')]
            with (a.output / f'{method}_s0.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                status['child_pid'] = child.pid
                while child.poll() is None:
                    status['heartbeat'] = time.time()
                    atomic(a.output / 'pair_status.json', status)
                    time.sleep(10)
                assert child.returncode == 0, f'{method} failed with exit {child.returncode}'
            status['completed_models'] += 1
        summary = {}
        for domain in ['d0', 'd1', 'd2']:
            scores = {m: json.loads((a.output / f'{m}_s0' / domain / 'scores.json').read_text())
                      for m in ['dp', 'equidiff']}
            assert scores['dp']['scenes'] == scores['equidiff']['scenes'], f'Unpaired scenes: {domain}'
            summary[domain] = {m: scores[m]['success_rate'] for m in scores}
        atomic(a.output / 'paired_results.json', {'state': 'complete', 'training_seed': 0,
            'protocol': status['protocol'], 'all_300_scenes_paired': True, 'domains': summary,
            'scope': 'Single training seed; no cross-seed superiority claim'})
        status.update(state='complete', completed_at=time.time(), child_pid=None)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), heartbeat=time.time())
        raise
    finally:
        atomic(a.output / 'pair_status.json', status)
