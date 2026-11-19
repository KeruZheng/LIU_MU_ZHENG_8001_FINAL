import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from pusht_study import ROOT
from pusht_study.train import train


class ResumeTests(unittest.TestCase):
    def test_interrupted_run_matches_uninterrupted_weights(self):
        if not (ROOT / "data/pusht/pusht_cchi_v7_replay.zarr").exists():
            self.skipTest("Official dataset required for integration test")
        config = json.loads((ROOT / "configs/smoke_cpu.json").read_text())
        config.update(steps=4, cpu_threads=2)
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            uninterrupted = train(config, Path(temp) / "full", "cpu")
            train({**config, "steps": 2}, Path(temp) / "resumed", "cpu")
            resumed = train(config, Path(temp) / "resumed", "cpu", resume=True)
            left = torch.load(uninterrupted, weights_only=False)
            right = torch.load(resumed, weights_only=False)
            self.assertEqual(left["step"], right["step"])
            for group in ("model", "ema"):
                for name, weight in left[group].items():
                    torch.testing.assert_close(weight, right[group][name], rtol=0, atol=0)
            torch.testing.assert_close(left["data_rng"], right["data_rng"], rtol=0, atol=0)
