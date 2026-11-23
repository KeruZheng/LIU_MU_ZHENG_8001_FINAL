import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from recovery_dispatch import main


class RecoveryTests(unittest.TestCase):
    def exercise(self, diagnostic, cap_once=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / 'fleet.json'
            config.write_text(json.dumps({'servers': [{'name': 'sicc', 'root': '/project', 'ssh': ['ssh', 'server']}],
                'services': [{'name': 'E4 recovery dispatcher'}, {'name': 'other_monitor'}],
                'auxiliary': [{'job_id': '29342'}]}))
            submissions = []
            record = {'job_id': '40000', 'release': 'e4_square_d0_s2_sicc_20261123_v2'}

            def run(command, **kwargs):
                line = command[-1]
                code, output, error = 0, '', ''
                if line.startswith('sacct '):
                    output = diagnostic + '|0:0\n'
                elif 'egl_diagnostic_v2.json' in line:
                    output = '{"state": "passed"}'
                elif 'retry_egl.py' in line:
                    submissions.append(line)
                    if cap_once and len(submissions) == 1:
                        code, error = 1, 'sbatch: error: MaxSubmitJobsPerAccount'
                elif 'submitted_s2_v2.json' in line:
                    output = json.dumps(record)
                else:
                    self.fail('Unexpected command: ' + line)
                return subprocess.CompletedProcess(command, code, output, error)

            with patch.object(sys, 'argv', ['dispatch', '--fleet-config', str(config),
                    '--output', str(root / 'out'), '--diagnostic-job', '29411']), \
                 patch('recovery_dispatch.subprocess.run', side_effect=run), \
                 patch('recovery_dispatch.time.sleep'):
                main()
            status = json.loads((root / 'out/recovery_status.json').read_text())
            changed = json.loads(config.read_text())
            return status, changed, submissions

    def test_failed_graphics_never_submits_training(self):
        status, _, submissions = self.exercise('FAILED')
        self.assertEqual(status['state'], 'failed_diagnostic')
        self.assertEqual(submissions, [])

    def test_account_cap_waits_then_registers_success(self):
        status, config, submissions = self.exercise('COMPLETED', cap_once=True)
        self.assertEqual(status['state'], 'submitted')
        self.assertEqual(len(submissions), 2)
        self.assertEqual(config['services'], [{'name': 'other_monitor'}])
        self.assertEqual(config['auxiliary'][-1]['job_id'], '40000')


if __name__ == '__main__':
    unittest.main()
