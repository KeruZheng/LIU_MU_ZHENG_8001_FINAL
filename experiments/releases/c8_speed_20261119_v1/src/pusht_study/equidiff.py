"""Push-T adaptation of the pinned EquiDiff architecture (not Frame Averaging)."""

import sys
import torch
from . import ROOT

sys.path.insert(0, str(ROOT / "upstream/equidiff"))
from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup
from equi_diffpo.model.equi.equi_encoder import EquiResBlock
from equi_diffpo.model.equi.equi_conditional_unet1d import EquiDiffusionUNet


class PlanarEquiUNet(EquiDiffusionUNet):
    def getOutFieldType(self):
        return enn.FieldType(self.group, [self.group.irrep(1)])

    def getActionGeometricTensor(self, act):
        # escnn rho1 uses y-up; Push-T image/world coordinates use y-down.
        return enn.GeometricTensor(act * act.new_tensor([1, -1]), self.getOutFieldType())

    def getOutput(self, out):
        return out * out.new_tensor([1, -1])


class EquiResEncoder96(torch.nn.Module):
    def __init__(self, hidden=128, order=4):
        super().__init__()
        group = gspaces.rot2dOnR2(order)
        self.in_type = enn.FieldType(group, 3 * [group.trivial_repr])
        channels = [hidden // 8, hidden // 4, hidden // 2, hidden]
        types = [enn.FieldType(group, c * [group.regular_repr]) for c in channels]
        layers = [enn.R2Conv(self.in_type, types[0], 5, padding=2), enn.ReLU(types[0])]
        for i, width in enumerate(channels):
            previous = channels[max(0, i - 1)]
            layers += [EquiResBlock(group, previous, width), EquiResBlock(group, width, width),
                       enn.PointwiseMaxPool(types[i], 2 if i < 3 else 3)]
        # 96 -> 48 -> 24 -> 12 -> 4 -> 1. No odd-size pooling discards a border.
        layers += [enn.R2Conv(types[-1], types[-1], 4), enn.ReLU(types[-1])]
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, image):
        if image.shape[-2:] != (96, 96):
            raise ValueError("EquiDiff Push-T encoder requires 96x96 images")
        return self.conv(enn.GeometricTensor(image, self.in_type)).tensor.flatten(1)


class EquiDiffPlanar(torch.nn.Module):
    uses_escnn = True

    def __init__(self, config):
        super().__init__()
        tiny = config.get("tiny", False)
        self.obs_steps = config["obs_steps"]
        self.order = 8 if config["method"] == "equidiff_c8" else 4
        hidden = config.get("equi_hidden", 8 if tiny else 128)
        if hidden < 8 or hidden % 8:
            raise ValueError("equi_hidden must be a positive multiple of 8")
        group = gspaces.no_base_space(CyclicGroup(self.order))
        self.encoder = EquiResEncoder96(hidden, self.order)
        self.obs_type = enn.FieldType(group, hidden * [group.regular_repr] + [group.irrep(1)])
        self.obs_projection = enn.Linear(self.obs_type,
                                        enn.FieldType(group, hidden * [group.regular_repr]))
        self.diff = PlanarEquiUNet(
            act_emb_dim=config.get("act_emb_dim", 8 if tiny else 64),
            local_cond_dim=None, global_cond_dim=self.obs_steps * hidden,
            diffusion_step_embed_dim=32 if tiny else 128,
            down_dims=(16, 32) if tiny else config["down_dims"], kernel_size=5,
            n_groups=8, cond_predict_scale=True, N=self.order)
        self.train()  # Discard escnn's initial inference caches before checkpointing.

    def encode(self, image, agent_pos):
        batch, steps = image.shape[:2]
        if steps != self.obs_steps:
            raise ValueError("Observation history differs from configuration")
        encoded = self.encoder(image.flatten(0, 1))
        position = agent_pos.flatten(0, 1) * agent_pos.new_tensor([1, -1])
        combined = enn.GeometricTensor(torch.cat((encoded, position), -1), self.obs_type)
        return self.obs_projection(combined).tensor.reshape(batch, -1)

    def predict_encoded(self, sample, timestep, encoded):
        return self.diff(sample, timestep, global_cond=encoded)

    def forward(self, sample, timestep, image, agent_pos):
        return self.predict_encoded(sample, timestep, self.encode(image, agent_pos))
