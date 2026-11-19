import unittest
import numpy as np
from pusht_study.statistics import paired_bootstrap, evaluation_array


class StatisticsTests(unittest.TestCase):
    def test_paired_constant_gap_has_exact_interval(self):
        right = np.random.default_rng(2).uniform(0, 0.7, size=(3, 25, 4))
        result = paired_bootstrap(right + 0.2, right, repetitions=1000)
        np.testing.assert_allclose(result["ci95"], [0.2, 0.2], atol=1e-14)

    def test_training_seed_uncertainty_is_not_erased_by_episodes(self):
        left = np.broadcast_to(np.array([0., 0.5, 1.])[:, None, None], (3, 25, 4))
        result = paired_bootstrap(left, np.zeros_like(left), repetitions=5000)
        self.assertLess(result["ci95"][0], 0.1)
        self.assertGreater(result["ci95"][1], 0.9)

    def test_incomplete_orientation_grid_rejected(self):
        with self.assertRaises(ValueError):
            evaluation_array([{"training_seed": 0, "state": "complete", "episode_results": [
                {"base_seed": 100000, "orientation": 0, "max_coverage": 0.2}]}])
