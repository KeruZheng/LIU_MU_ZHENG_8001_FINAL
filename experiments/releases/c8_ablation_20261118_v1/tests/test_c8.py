import unittest
import torch
from pusht_study.models import make_model
from pusht_study.geometry import rotate_image, rotate_xy


class C8Tests(unittest.TestCase):
    def test_c8_denoiser_preserves_exact_c4_subgroup(self):
        torch.manual_seed(131)
        torch.set_num_threads(2)
        model = make_model(dict(method="equidiff_c8", obs_steps=2, down_dims=[16,32], tiny=True))
        image = torch.randn(1,2,3,96,96)
        position = torch.randn(1,2,2)
        action = torch.randn(1,16,2)
        t = torch.tensor([2])
        expected = model(action, t, image, position)
        for k in (1,2,3):
            actual = model(rotate_xy(action,k), t, rotate_image(image,k), rotate_xy(position,k))
            torch.testing.assert_close(actual, rotate_xy(expected,k), rtol=3e-4, atol=3e-5)
        expected.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_coordinate_clipping_is_not_a_c8_symmetry(self):
        # Prevent interpreting the existing square-domain DDPM clip as SO(2)/C8 equivariant.
        x = torch.tensor([[1.5, 0.]])
        c = 2 ** -0.5
        rotation = torch.tensor([[c, -c], [c, c]])
        left = (x @ rotation).clamp(-1, 1)
        right = x.clamp(-1, 1) @ rotation
        self.assertGreater((left - right).abs().max().item(), 0.1)
