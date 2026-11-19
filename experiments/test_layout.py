import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_run


class LayoutTests(unittest.TestCase):
    def test_all_published_files_match_their_snapshot(self):
        for p in (prepare_run.CODES / 'experiments/releases').glob('*/SNAPSHOT.json'):
            for name, expected in json.loads(p.read_text())['files'].items():
                self.assertEqual(hashlib.sha256((p.parent / name).read_bytes()).hexdigest(), expected, str(p.parent / name))

    def test_reproduction_stays_in_task_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'experiments/releases/example'
            (source / 'configs').mkdir(parents=True)
            config = source / 'configs/run0.json'; config.write_text('{}')
            (source / 'SNAPSHOT.json').write_text(json.dumps({'files': {'configs/run0.json': hashlib.sha256(config.read_bytes()).hexdigest()}}))
            task = root / 'tasks/rq1_transfer'; task.mkdir(parents=True)
            (task / 'task.json').write_text(json.dumps({'releases': ['example']}))
            (root / 'data').mkdir(); (root / 'upstream').mkdir()
            with patch.object(prepare_run, 'CODES', root):
                target = prepare_run.prepare('rq1_transfer', 'example', 'run0', 'trial', root / 'data', root / 'upstream')
                self.assertEqual(target, task / 'runs/trial')
                self.assertTrue((target / 'data').is_symlink())
                with self.assertRaises(FileExistsError):
                    prepare_run.prepare('rq1_transfer', 'example', 'run0', 'trial', root / 'data', root / 'upstream')
                config.write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, 'Changed release'):
                    prepare_run.prepare('rq1_transfer', 'example', 'run0', 'different', root / 'data', root / 'upstream')


if __name__ == '__main__':
    unittest.main()
