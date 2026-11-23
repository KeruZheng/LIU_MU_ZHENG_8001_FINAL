"""Exercise supervisor replacement with real disposable processes, never a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from local_owner import identity, same_live


TRAINER = '''
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser()
for key in ('root','codes','method','seed','mode'): p.add_argument('--'+key)
a=p.parse_args(); root=Path(a.root); name=a.method+'_square_d0_s0'
out=root/'runs'/name; out.mkdir(exist_ok=True)
if a.method=='equidiff':
    while not (root/'finish').exists(): time.sleep(0.05)
for d in ('d0','d1','d2'):
    target=out/('transfer_'+d); target.mkdir()
    (target/'scores.json').write_text(json.dumps({'episodes':100}))
(out/'local_inventory.json').write_text('{}')
(out/'completed.json').write_text(json.dumps({'state':'complete','epoch':500,'global_step':112000,'target_updates':112000}))
'''

SUPERVISOR = '''
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
r=Path(sys.argv[1])
with (r/'results/compute.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    with (r/'logs/equidiff.log').open('w') as log:
        child=subprocess.Popen([sys.executable,str(r/'run.py'),'--root',str(r),'--seed','0','--method','equidiff'],stdout=log,stderr=log)
        (r/'results/supervisor_status.json').write_text(json.dumps({'supervisor_pid':os.getpid(),'child_pid':child.pid,'run':'equidiff_square_d0_s0','state':'training_and_transfer'}))
        child.wait()
'''


def wait_for(predicate, seconds=25):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.05)
    raise TimeoutError('Disposable process test timed out')


class HandoffTests(unittest.TestCase):
    def test_active_child_survives_and_only_seed_zero_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('results', 'logs', 'runs/equidiff_square_d0_s0'):
                (root / name).mkdir(parents=True)
            (root / 'run.py').write_text(TRAINER)
            (root / 'supervise.py').write_text(SUPERVISOR)
            (root / 'frozen_files.json').write_text('{}')
            (root / 'runs/equidiff_square_d0_s0/status.json').write_text('{"epoch": 1}')
            parent = subprocess.Popen([sys.executable, str(root / 'supervise.py'), str(root)])
            owner = None
            try:
                wait_for(lambda: (root / 'results/supervisor_status.json').exists())
                state = json.loads((root / 'results/supervisor_status.json').read_text())
                child = state['child_pid']
                original = identity(child)
                owner = subprocess.Popen([sys.executable, str(Path(__file__).with_name('local_owner.py')),
                    '--root', str(root), '--codes', str(root), '--old-pid', str(parent.pid),
                    '--child-pid', str(child), '--receipt', str(root / 'handoff.json')])
                def adopted():
                    p = root / 'handoff.json'
                    return p.exists() and json.loads(p.read_text())['state'] == 'adopted'
                wait_for(adopted)
                self.assertTrue(same_live(child, original))
                self.assertEqual(parent.wait(timeout=5), -15)
                (root / 'finish').touch()
                self.assertEqual(owner.wait(timeout=30), 0)
                final = json.loads((root / 'results/supervisor_status.json').read_text())
                self.assertEqual(final['state'], 'complete')
                self.assertEqual(final['completed_runs'], 2)
                self.assertEqual(sorted(p.name for p in (root / 'runs').iterdir()),
                                 ['dp_square_d0_s0', 'equidiff_square_d0_s0'])
            finally:
                (root / 'finish').touch()
                for proc in (owner, parent):
                    if proc and proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
