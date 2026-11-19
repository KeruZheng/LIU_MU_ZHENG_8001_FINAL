import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import zarr

from pusht_study.geometry import rotate_xy, rotate_image, normalize_xy, WORLD_CENTER
from pusht_study.data import make_split, PushTSequences
from pusht_study.models import Denoiser, C4FrameAverage


class GeometryTests(unittest.TestCase):
    def test_pixel_and_coordinate_rotation_agree(self):
        image = np.zeros((9, 9))
        image[2, 6] = 1
        for k in range(4):
            xy = rotate_xy(np.array([6., 2.]), k, 4)
            actual = np.argwhere(rotate_image(image, k) == 1)[0]
            np.testing.assert_array_equal(actual, xy[::-1])

    def test_group_composition_and_normalization(self):
        x = np.array([[50., 400.], [255.5, 300.]])
        for g in range(4):
            for h in range(4):
                np.testing.assert_allclose(rotate_xy(rotate_xy(x, g, WORLD_CENTER), h, WORLD_CENTER),
                                           rotate_xy(x, g + h, WORLD_CENTER))
            np.testing.assert_allclose(normalize_xy(rotate_xy(x, g, WORLD_CENTER)),
                                       rotate_xy(normalize_xy(x), g))


class DatasetTests(unittest.TestCase):
    def test_disjoint_nested_balanced_split(self):
        split = make_split(206)
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertEqual(len(split["train"]), 200)
        counts = np.bincount([split["orientation"][str(i)] for i in split["train"][:50]], minlength=4)
        self.assertLessEqual(counts.max() - counts.min(), 1)
        self.assertEqual(split, make_split(206))

    def test_sequences_never_cross_episode_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root = zarr.open_group(temp, mode="w")
            root.create_dataset("meta/episode_ends", data=np.array([20, 40, 60]))
            labels = np.repeat(np.arange(3), 20)
            root.create_dataset("data/img", data=np.zeros((60, 8, 8, 3), dtype=np.uint8))
            root.create_dataset("data/state", data=np.repeat(labels[:, None], 5, axis=1).astype(float))
            root.create_dataset("data/action", data=np.repeat(labels[:, None], 2, axis=1).astype(float))
            split = make_split(3, val_count=1)
            dataset = PushTSequences(Path(temp), split, demos=2)
            for index in range(len(dataset)):
                row = dataset[index]
                expected = normalize_xy(float(row["episode"]))
                np.testing.assert_allclose(row["action"].numpy(), expected)

    def test_rotated_dataset_applies_fixed_base_orientation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = zarr.open_group(temp, mode="w")
            root.create_dataset("meta/episode_ends", data=np.array([20, 40, 60, 80]))
            img = np.zeros((80, 8, 8, 3), dtype=np.uint8)
            state = np.zeros((80, 5), dtype=np.float32)
            action = np.zeros((80, 2), dtype=np.float32)
            for episode in range(4):
                sl = slice(episode * 20, (episode + 1) * 20)
                img[sl, 1, 2, :] = episode + 1
                state[sl, 0] = 100 + 20 * episode
                state[sl, 1] = 200 + 10 * episode
                action[sl, 0] = 300 + 5 * episode
                action[sl, 1] = 150 - 5 * episode
            root.create_dataset("data/img", data=img)
            root.create_dataset("data/state", data=state)
            root.create_dataset("data/action", data=action)
            split = make_split(4, val_count=1)
            canonical = PushTSequences(Path(temp), split, demos=3, rotated=False)
            rotated = PushTSequences(Path(temp), split, demos=3, rotated=True)
            index = next(i for i, (episode, *_rest) in enumerate(rotated.indices)
                         if rotated.orientation[str(episode)] != 0)
            k = rotated.orientation[str(rotated.indices[index][0])]
            a, b = canonical[index], rotated[index]
            torch.testing.assert_close(b["image"], rotate_image(a["image"], k))
            torch.testing.assert_close(b["agent_pos"], rotate_xy(a["agent_pos"], k))
            torch.testing.assert_close(b["action"], rotate_xy(a["action"], k))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_frame_average_is_equivariant_and_differentiable(self):
        torch.manual_seed(11)
        model = C4FrameAverage(Denoiser(tiny=True))
        image, pos = torch.randn(2, 2, 3, 32, 32), torch.randn(2, 2, 2)
        action, t = torch.randn(2, 16, 2), torch.tensor([1, 3])
        y = model(action, t, image, pos)
        for k in range(4):
            actual = model(rotate_xy(action, k), t, rotate_image(image, k), rotate_xy(pos, k))
            torch.testing.assert_close(actual, rotate_xy(y, k), atol=2e-6, rtol=2e-5)
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
