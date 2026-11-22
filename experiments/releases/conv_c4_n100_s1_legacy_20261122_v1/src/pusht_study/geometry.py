"""C4 action aligned with np/torch.rot90 on images (x right, y down)."""

import numpy as np
import torch

WORLD_CENTER = 255.5
WORLD_SCALE = 255.5


def rotate_xy(x, k, center=0.0):
    if x.shape[-1] != 2:
        raise ValueError("Expected (..., 2) coordinates")
    v = x - center
    a, b = v[..., 0], v[..., 1]
    pairs = ((a, b), (b, -a), (-a, -b), (-b, a))
    values = pairs[int(k) % 4]
    out = torch.stack(values, dim=-1) if torch.is_tensor(x) else np.stack(values, axis=-1)
    return out + center


def rotate_image(image, k):
    if image.shape[-2] != image.shape[-1]:
        raise ValueError("C4 image rotations require square images in (..., H, W)")
    if torch.is_tensor(image):
        return torch.rot90(image, int(k) % 4, (-2, -1))
    return np.rot90(image, int(k) % 4, axes=(-2, -1)).copy()


def rotate_state(state, k):
    result = np.array(state, dtype=np.float64, copy=True)
    result[..., :2] = rotate_xy(result[..., :2], k, WORLD_CENTER)
    result[..., 2:4] = rotate_xy(result[..., 2:4], k, WORLD_CENTER)
    result[..., 4] = (result[..., 4] - int(k) * np.pi / 2) % (2 * np.pi)
    return result


def normalize_xy(x):
    return (x - WORLD_CENTER) / WORLD_SCALE


def denormalize_xy(x):
    return x * WORLD_SCALE + WORLD_CENTER
