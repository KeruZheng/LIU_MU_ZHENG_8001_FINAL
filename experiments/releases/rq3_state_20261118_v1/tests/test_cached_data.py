import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from pusht_study import ROOT
from pusht_study.data import PushTSequences
from pusht_study.train import train


class CachedDataTests(unittest.TestCase):
    def test_cached_batches_match_zarr_at_episode_boundaries(self):
        split = json.loads((ROOT / "manifests/split.json").read_text())
        path = ROOT / "data/pusht/pusht_cchi_v7_replay.zarr"
        for rotated in (False, True):
            dataset = PushTSequences(path, split, demos=4, rotated=rotated, cache=True)
            indices = [i for i, row in enumerate(dataset.indices)
                       if i == 0 or row[0] != dataset.indices[i - 1][0]]
            indices += [i - 1 for i in indices if i] + [len(dataset) - 1]
            batch = dataset.get_batch(indices)
            for key, actual in batch.items():
                expected = torch.stack([dataset[i][key] for i in indices])
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_cache_preserves_training_and_retains_ema(self):
        config = json.loads((ROOT / "configs/smoke_cpu.json").read_text())
        config.update(method="dp_c4_aug", steps=4, cpu_threads=2, rotated=True,
                      allow_rotated=True, retain_ema_steps=[2, 4])
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            first = train(config, Path(temp) / "zarr", "cpu")
            second = train({**config, "cache_data": True}, Path(temp) / "cache", "cpu")
            left = torch.load(first, weights_only=False)
            right = torch.load(second, weights_only=False)
            for group in ("model", "ema"):
                for name, weight in left[group].items():
                    torch.testing.assert_close(weight, right[group][name], rtol=0, atol=0)
            saved = torch.load(Path(temp) / "cache/ema_step_4.pt", weights_only=False)
            self.assertEqual(saved["step"], 4)
            self.assertNotIn("optimizer", saved)
            for name, weight in saved["ema"].items():
                torch.testing.assert_close(weight, right["ema"][name], rtol=0, atol=0)

    def test_vectorized_rotation_draws_preserve_rng_stream(self):
        first = torch.Generator().manual_seed(33)
        second = torch.Generator().manual_seed(33)
        expected = torch.stack([torch.randint(4, (), generator=first) for _ in range(64)])
        torch.testing.assert_close(torch.randint(4, (64,), generator=second), expected)
        torch.testing.assert_close(first.get_state(), second.get_state())
