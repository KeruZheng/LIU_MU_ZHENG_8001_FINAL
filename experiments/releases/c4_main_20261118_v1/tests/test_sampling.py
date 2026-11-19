import unittest
from unittest.mock import patch

import torch
from diffusers import DDPMScheduler
from pusht_study.geometry import rotate_image, rotate_xy
from pusht_study.models import Denoiser, C4FrameAverage


class SamplingTests(unittest.TestCase):
    def test_ddpm_chain_with_coupled_noise_is_equivariant(self):
        torch.set_num_threads(2)
        torch.manual_seed(91)
        model = C4FrameAverage(Denoiser(tiny=True)).eval()
        image = torch.randn(1, 2, 3, 32, 32)
        pos = torch.randn(1, 2, 2)
        initial = torch.randn(1, 16, 2)
        noises = [torch.randn_like(initial) for _ in range(3)]

        @torch.no_grad()
        def sample(k):
            scheduler = DDPMScheduler(num_train_timesteps=4, beta_schedule="squaredcos_cap_v2")
            scheduler.set_timesteps(4)
            state = rotate_xy(initial, k)
            encoded = model.encode(rotate_image(image, k), rotate_xy(pos, k))
            with patch("diffusers.schedulers.scheduling_ddpm.randn_tensor",
                       side_effect=[rotate_xy(n, k) for n in noises]):
                for t in scheduler.timesteps:
                    state = scheduler.step(model.predict_encoded(state, t, encoded), t, state).prev_sample
            return state

        base = sample(0)
        for k in range(1, 4):
            torch.testing.assert_close(sample(k), rotate_xy(base, k), atol=2e-5, rtol=2e-5)
