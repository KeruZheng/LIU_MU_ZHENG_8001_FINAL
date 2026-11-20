import unittest
import numpy as np
import torch
from pusht_study.models import make_model
from pusht_study.geometry import rotate_xy, rotate_image
from pusht_study.env_c4 import PushTC4Env


class ImageRQ3Tests(unittest.TestCase):
    def test_zero_gate_equivariance_gradient_and_nonzero_breaking(self):
        torch.set_num_threads(2)
        torch.manual_seed(812)
        model = make_model(dict(method='equidiff_c4_gate', tiny=True, obs_steps=2, down_dims=[16, 32]))
        image = torch.randn(2, 2, 3, 96, 96)
        position = torch.randn(2, 2, 2)
        action = torch.randn(2, 16, 2)
        timestep = torch.tensor([1, 3])
        expected = model(action, timestep, image, position)
        for k in (1, 2, 3):
            actual = model(rotate_xy(action, k), timestep, rotate_image(image, k), rotate_xy(position, k))
            torch.testing.assert_close(actual, rotate_xy(expected, k), rtol=3e-4, atol=3e-5)
        expected.square().mean().backward()
        self.assertGreater(abs(model.symmetry_gate.grad.item()), 1e-8)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with torch.no_grad():
            model.symmetry_gate.fill_(1.)
            expected = model(action, timestep, image, position)
            actual = model(rotate_xy(action, 1), timestep, rotate_image(image, 1), rotate_xy(position, 1))
            self.assertGreater((actual - rotate_xy(expected, 1)).abs().max().item(), 1e-5)

    def test_goal_visible_in_image_but_dynamics_identical(self):
        for k in (1, 2, 3):
            envs = [PushTC4Env(orientation=k, goal_mode=mode, legacy=True) for mode in ('fixed', 'rotated')]
            try:
                for env in envs:
                    env.seed(900001)
                observations = [env.reset() for env in envs]
                for _ in range(3):
                    np.testing.assert_allclose(observations[0]['state'], observations[1]['state'])
                    self.assertGreater(np.abs(observations[0]['image'] - observations[1]['image']).sum(), 0)
                    np.testing.assert_allclose(envs[0].goal_pose, [256, 256, np.pi / 4])
                    observations = [env.step(np.array([240., 250.]))[0] for env in envs]
            finally:
                for env in envs:
                    env.close()


if __name__ == '__main__':
    unittest.main()
