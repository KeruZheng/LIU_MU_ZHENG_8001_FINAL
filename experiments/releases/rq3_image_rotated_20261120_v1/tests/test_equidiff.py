import unittest
import torch
from diffusers import DDPMScheduler
from unittest.mock import patch

from pusht_study.models import make_model
from pusht_study.geometry import rotate_xy, rotate_image


class EquiDiffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = make_model(dict(method="equidiff_c4", obs_steps=2, tiny=True,
                                    down_dims=[16, 32])).eval()

    def test_denoiser_equivariance_and_backward(self):
        image = torch.randn(2, 2, 3, 96, 96)
        agent = torch.randn(2, 2, 2)
        action = torch.randn(2, 16, 2)
        timestep = torch.tensor([1, 3])
        expected = self.model(action, timestep, image, agent)
        for k in range(1, 4):
            actual = self.model(rotate_xy(action, k), timestep, rotate_image(image, k), rotate_xy(agent, k))
            torch.testing.assert_close(actual, rotate_xy(expected, k), rtol=3e-4, atol=3e-5)
        expected.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))

    def test_coupled_ddpm_sampler(self):
        image = torch.randn(1, 2, 3, 96, 96)
        agent = torch.randn(1, 2, 2)
        initial = torch.randn(1, 16, 2)
        noises = [torch.randn_like(initial) for _ in range(3)]

        @torch.no_grad()
        def sample(k):
            scheduler = DDPMScheduler(num_train_timesteps=4, beta_schedule="squaredcos_cap_v2")
            scheduler.set_timesteps(4)
            state = rotate_xy(initial, k)
            encoded = self.model.encode(rotate_image(image, k), rotate_xy(agent, k))
            with patch("diffusers.schedulers.scheduling_ddpm.randn_tensor",
                       side_effect=[rotate_xy(n, k) for n in noises]):
                for t in scheduler.timesteps:
                    state = scheduler.step(self.model.predict_encoded(state, t, encoded), t, state).prev_sample
            return state

        expected = sample(0)
        for k in range(1, 4):
            torch.testing.assert_close(sample(k), rotate_xy(expected, k), rtol=5e-4, atol=5e-5)
