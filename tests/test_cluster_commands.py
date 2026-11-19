import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class ClusterPreviewTests(unittest.TestCase):
    def test_all_actions_use_configured_user_without_execution(self):
        path = Path(__file__).resolve().parents[1] / 'scripts/cluster.py'
        spec = importlib.util.spec_from_file_location('cluster_preview', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / 'configs').mkdir()
            cfg = dict(host='alice@example.invalid', root='/project/robotic', ssh_options=[],
                       group=None, partition='gpu', account=None, qos=None, cpus=4, memory='16G')
            (root / 'configs/clusters.local.json').write_text(json.dumps({'sicc': cfg}))
            for action in ('check', 'upload', 'preflight', 'submit'):
                argv = ['cluster.py', action]
                if action == 'submit':
                    argv += ['--config', 'configs/example.json', '--run', 'example']
                with patch.object(module, 'ROOT', root), patch('sys.argv', argv), \
                     patch.object(module.subprocess, 'run') as execute, \
                     contextlib.redirect_stdout(io.StringIO()) as output:
                    module.main()
                    self.assertIn('squeue -u alice', output.getvalue())
                    execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
