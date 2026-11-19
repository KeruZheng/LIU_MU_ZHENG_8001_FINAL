import json
import unittest
import numpy as np
import torch
from pusht_study import ROOT
from pusht_study.data import PushTStateSequences
from pusht_study.env_c4 import PushTC4Env
from pusht_study.geometry import rotate_xy
from pusht_study.models import make_model


class StateRQ3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(812)
        self.state = torch.randn(2, 2, 6)
        self.goal = torch.randn(2, 2, 2)
        self.action = torch.randn(2, 16, 2)
        self.t = torch.tensor([1, 3])

    def model(self, method):
        return make_model(dict(method=method, observation="state", tiny=True, obs_steps=2, down_dims=[16,32]))

    def rotated_state(self, k):
        return rotate_xy(self.state.reshape(2, 2, 3, 2), k).reshape(2, 2, 6)

    def test_goal_conditioned_equivariance_and_zero_gate(self):
        for method in ("state_equidiff", "state_equidiff_goal", "state_equidiff_gate"):
            model = self.model(method)
            expected = model(self.action, self.t, self.state, self.goal)
            for k in (1, 2, 3):
                actual = model(rotate_xy(self.action, k), self.t, self.rotated_state(k), rotate_xy(self.goal, k))
                torch.testing.assert_close(actual, rotate_xy(expected, k), rtol=2e-4, atol=2e-5)

    def test_gate_receives_gradient_and_can_break_equivariance(self):
        model = self.model("state_equidiff_gate")
        model(self.action, self.t, self.state, self.goal).square().mean().backward()
        self.assertGreater(abs(model.symmetry_gate.grad.item()), 1e-8)
        with torch.no_grad():
            model.symmetry_gate.fill_(1.)
            base = model(self.action, self.t, self.state, self.goal)
            actual = model(rotate_xy(self.action, 1), self.t, self.rotated_state(1), rotate_xy(self.goal, 1))
            self.assertGreater((actual - rotate_xy(base, 1)).abs().max().item(), 1e-5)

    def test_goal_is_hidden_from_plain_models(self):
        for method in ("state_dp", "state_equidiff", "state_equidiff_gate"):
            model = self.model(method)
            left = model(self.action, self.t, self.state, self.goal)
            right = model(self.action, self.t, self.state, self.goal * 10)
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        model = self.model("state_equidiff_goal")
        left = model(self.action, self.t, self.state, self.goal)
        right = model(self.action, self.t, self.state, self.goal * 10)
        self.assertGreater((left - right).abs().max().item(), 1e-5)

    def test_state_data_rotates_all_vector_features_and_goal(self):
        split = json.loads((ROOT / "manifests/split.json").read_text())
        path = ROOT / "data/pusht/pusht_cchi_v7_replay.zarr"
        canonical = PushTStateSequences(path, split, demos=4)
        rotated = PushTStateSequences(path, split, demos=4, rotated=True)
        for k in range(4):
            i = next(i for i, row in enumerate(canonical.indices) if split["orientation"][str(row[0])] == k)
            a, b = canonical.get_batch([i]), rotated.get_batch([i])
            torch.testing.assert_close(b["state"].reshape(1,2,3,2), rotate_xy(a["state"].reshape(1,2,3,2), k), rtol=0, atol=0)
            torch.testing.assert_close(b["goal"], rotate_xy(a["goal"], k), rtol=0, atol=0)
            torch.testing.assert_close(b["action"], rotate_xy(a["action"], k), rtol=0, atol=0)

    def test_fixed_goal_is_not_rotated_or_leaked_into_state(self):
        for k in range(4):
            envs = [PushTC4Env(orientation=k, goal_mode=mode, render_observations=False, legacy=True)
                    for mode in ("fixed", "rotated")]
            try:
                for env in envs:
                    env.seed(900001)
                fixed, rotated = [env.reset() for env in envs]
                np.testing.assert_array_equal(fixed["state"], rotated["state"])
                self.assertNotIn("image", fixed)
                np.testing.assert_allclose(envs[0].goal_pose, [256,256,np.pi/4])
                np.testing.assert_allclose(rotated["goal"], rotate_xy(fixed["goal"], k), atol=1e-7)
            finally:
                for env in envs:
                    env.close()
