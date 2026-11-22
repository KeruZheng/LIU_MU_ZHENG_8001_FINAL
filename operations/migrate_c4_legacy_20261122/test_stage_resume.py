import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from stage_resume import REMOTE_VERIFY


class StagingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.run = self.root / "releases/test/runs/equidiff_c4_n25_s0"
        self.run.mkdir(parents=True)
        self.partial = self.run / "import_last.pt.part"
        self.partial.write_bytes(b"verified checkpoint")
        self.request = dict(remote_root=str(self.root), release="test", job_id="123",
                            bytes=self.partial.stat().st_size,
                            sha256=hashlib.sha256(self.partial.read_bytes()).hexdigest())

    def execute(self, description):
        with patch("sys.stdin", io.StringIO(json.dumps(self.request))), \
                patch("subprocess.check_output", return_value=description), \
                patch("subprocess.run") as release, contextlib.redirect_stdout(io.StringIO()):
            exec(REMOTE_VERIFY, {})
        return release

    def test_verified_checkpoint_releases_only_named_held_job(self):
        release = self.execute("JobName=robotic_c4_mig_b JobState=PENDING Reason=JobHeldUser ")
        release.assert_called_once_with(["scontrol", "release", "123"], check=True)
        self.assertFalse(self.partial.exists())
        self.assertTrue((self.run / "last.pt").exists())
        self.assertEqual(json.loads((self.run / "checkpoint_import_verified.json").read_text())["state"], "verified")

    def test_bad_hash_keeps_job_held_and_input_partial(self):
        self.request["sha256"] = "wrong"
        with patch("subprocess.run") as release:
            with self.assertRaises(AssertionError):
                self.execute("JobName=robotic_c4_mig_b JobState=PENDING Reason=JobHeldUser ")
            release.assert_not_called()
        self.assertTrue(self.partial.exists())
        self.assertFalse((self.run / "last.pt").exists())

    def test_wrong_job_name_cannot_be_released(self):
        with self.assertRaises(AssertionError):
            self.execute("JobName=unrelated_job JobState=PENDING Reason=JobHeldUser ")

    def test_already_running_job_is_not_released_again(self):
        release = self.execute("JobName=robotic_c4_mig_b JobState=RUNNING Reason=None ")
        release.assert_not_called()


if __name__ == "__main__":
    unittest.main()
