"""DP-style baseline and explicit C4 Frame Averaging, not original EquiDiff."""

import torch
from torch import nn
from torchvision.models import resnet18
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from .geometry import rotate_image, rotate_xy


def group_norm(channels):
    return nn.GroupNorm(min(8, channels), channels)


class Denoiser(nn.Module):
    def __init__(self, obs_steps=2, down_dims=(256, 512, 1024), tiny=False):
        super().__init__()
        self.obs_steps = obs_steps
        if tiny:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 8, 5, stride=4, padding=2), nn.GroupNorm(4, 8),
                nn.Mish(), nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            feature_dim = 8
        else:
            self.encoder = resnet18(weights=None, norm_layer=group_norm)
            self.encoder.fc = nn.Identity()
            feature_dim = 512
        self.unet = ConditionalUnet1D(
            input_dim=2, global_cond_dim=obs_steps * (feature_dim + 2),
            diffusion_step_embed_dim=32 if tiny else 128,
            down_dims=(16, 32) if tiny else down_dims,
            kernel_size=5, n_groups=8, cond_predict_scale=True,
        )

    def encode(self, image, agent_pos):
        batch, steps = image.shape[:2]
        if steps != self.obs_steps:
            raise ValueError("Observation history differs from model configuration")
        features = self.encoder(image.flatten(0, 1)).reshape(batch, steps, -1)
        return torch.cat((features, agent_pos), dim=-1).flatten(1)

    def predict_encoded(self, sample, timestep, encoded):
        return self.unet(sample, timestep, global_cond=encoded)

    def forward(self, sample, timestep, image, agent_pos):
        return self.predict_encoded(sample, timestep, self.encode(image, agent_pos))


class C4FrameAverage(nn.Module):
    """Average inverse-transformed noise predictions over all four group elements."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def encode(self, image, agent_pos):
        return [self.base.encode(rotate_image(image, k), rotate_xy(agent_pos, k))
                for k in range(4)]

    def predict_encoded(self, sample, timestep, encoded):
        predictions = [rotate_xy(self.base.predict_encoded(
            rotate_xy(sample, k), timestep, encoded[k]), -k) for k in range(4)]
        return torch.stack(predictions).mean(0)

    def forward(self, sample, timestep, image, agent_pos):
        return self.predict_encoded(sample, timestep, self.encode(image, agent_pos))


def make_model(config):
    if config["method"] == "equidiff_c4":
        from .equidiff import EquiDiffC4
        return EquiDiffC4(config)
    base = Denoiser(config["obs_steps"], config["down_dims"], config.get("tiny", False))
    method = config["method"]
    if method == "dp_c4_frameavg":
        return C4FrameAverage(base)
    if method in ("dp", "dp_c4_aug"):
        return base
    raise ValueError(f"Unimplemented method: {method}")
