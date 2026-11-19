from collections import deque
import unittest
import numpy as np
import torch
from pusht_study.env_c4 import PushTC4Env
from pusht_study.env import coverage
from pusht_study.geometry import rotate_xy, rotate_image, WORLD_CENTER
from pusht_study.evaluate import predict
from pusht_study.evaluate_c4 import predict_batch, summarize
from pusht_study.models import Denoiser


class C4EvaluationTests(unittest.TestCase):
    def test_native_steps_preserve_rotated_state_and_images(self):
        envs = [PushTC4Env(orientation=k, legacy=True) for k in range(4)]
        try:
            obs = []
            for env in envs:
                env.seed(900002)
                obs.append(env.reset())
            for action in np.random.default_rng(123).uniform(0, 511, size=(30, 2)):
                obs = [env.step(rotate_xy(action, k, WORLD_CENTER))[0] for k, env in enumerate(envs)]
                for k in range(1, 4):
                    np.testing.assert_allclose(obs[k]["agent_pos"],
                                               rotate_xy(obs[0]["agent_pos"], k, WORLD_CENTER), atol=1e-8)
                    np.testing.assert_allclose(obs[k]["image"], rotate_image(obs[0]["image"], k), atol=1/255)
                    self.assertAlmostEqual(coverage(envs[0]), coverage(envs[k]), places=8)
        finally:
            for env in envs:
                env.close()

    def test_batch_sampling_matches_independent_generators(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        model = Denoiser(tiny=True).eval()
        config = dict(obs_steps=2, horizon=16, action_steps=8, diffusion_steps=4, inference_steps=4)
        rng = np.random.default_rng(40)
        histories = [deque([{"image": rng.random((3, 32, 32), dtype=np.float32),
                             "agent_pos": rng.uniform(30, 480, size=2)} for _ in range(2)]) for _ in range(3)]
        actual = predict_batch(model, config, histories, "cpu",
                               [torch.Generator().manual_seed(seed) for seed in (2, 3, 4)])
        for i, seed in enumerate((2, 3, 4)):
            expected = predict(model, config, histories[i], "cpu", torch.Generator().manual_seed(seed))
            np.testing.assert_allclose(actual[i], expected, atol=0.002, rtol=2e-5)

    def test_orientation_summary_keeps_success_and_coverage_distinct(self):
        rows = [{"orientation": 0, "max_coverage": 0.5, "success": False},
                {"orientation": 0, "max_coverage": 1.0, "success": True},
                {"orientation": 1, "max_coverage": 0.9, "success": False}]
        result = summarize(rows)
        self.assertEqual(result["per_orientation"]["0"]["mean_max_coverage"], 0.75)
        self.assertEqual(result["per_orientation"]["1"]["success_rate"], 0)
        self.assertAlmostEqual(result["success_rate"], 1/3)
