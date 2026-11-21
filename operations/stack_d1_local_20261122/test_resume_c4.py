import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import resume_c4


class ResumeGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stack = self.root / "stack"
        self.run = self.stack / "runs/stack_d1_n100_s0"
        (self.stack / "results").mkdir(parents=True)
        (self.run / "checkpoints").mkdir(parents=True)
        self.lane = self.root / "c4"
        (self.lane / "manifests").mkdir(parents=True)
        self.write(self.stack / "results/supervisor_status.json", {"state": "complete"})
        self.write(self.run / "completed.json", {"state": "complete", "epoch": 500})
        self.write(self.run / "final_evaluation.json", {"test/mean_score": 0.5})
        checkpoint = self.run / "checkpoints/final.ckpt"
        checkpoint.write_bytes(b"stack")
        self.write(self.run / "local_inventory.json", {"files": [{"name": checkpoint.name,
            "bytes": 5, "sha256": hashlib.sha256(b"stack").hexdigest()}]})
        preserved = self.lane / "last.pt"
        preserved.write_bytes(b"c4")
        (self.lane / "supervise.py").write_bytes(b"frozen")
        self.write(self.lane / "manifests/frozen_files.json",
                   {"supervise.py": hashlib.sha256(b"frozen").hexdigest()})
        self.pause = self.root / "pause.json"
        self.write(self.pause, {"checkpoint": str(preserved), "checkpoint_step": 12000,
            "sha256": hashlib.sha256(b"c4").hexdigest(),
            "resume_command": ["python", "supervise.py", "--root", str(self.lane)]})
        self.args = ["resume_c4.py", "--stack-root", str(self.stack),
                     "--pause-receipt", str(self.pause), "--output", str(self.root / "watcher")]

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def test_complete_verified_run_resumes_only_once(self):
        with patch("sys.argv", self.args), patch("resume_c4.subprocess.Popen", return_value=Mock(pid=123)) as start:
            resume_c4.main()
            resume_c4.main()
            start.assert_called_once()

    def test_corrupt_stack_checkpoint_prevents_resume(self):
        (self.run / "checkpoints/final.ckpt").write_bytes(b"wrong")
        with patch("sys.argv", self.args), patch("resume_c4.subprocess.Popen") as start:
            with self.assertRaises(AssertionError):
                resume_c4.main()
            start.assert_not_called()

    def test_failed_stack_prevents_resume(self):
        self.write(self.stack / "results/supervisor_status.json", {"state": "failed"})
        with patch("sys.argv", self.args), patch("resume_c4.subprocess.Popen") as start:
            with self.assertRaises(RuntimeError):
                resume_c4.main()
            start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
