"""Image-conditioned C4 model with a learnable fixed world reference."""
import torch
from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup
from .equidiff import EquiDiffC4


class ImageEquiGate(EquiDiffC4):
    def __init__(self, config):
        super().__init__(config)
        hidden = config.get('equi_hidden', 8 if config.get('tiny', False) else 128)
        group = gspaces.no_base_space(CyclicGroup(4))
        self.obs_type = enn.FieldType(group, hidden * [group.regular_repr] + 2 * [group.irrep(1)])
        self.obs_projection = enn.Linear(self.obs_type, enn.FieldType(group, hidden * [group.regular_repr]))
        self.symmetry_gate = torch.nn.Parameter(torch.zeros(()))
        self.train()

    def encode(self, image, agent_pos):
        batch, steps = image.shape[:2]
        if steps != self.obs_steps:
            raise ValueError('Observation history differs from configuration')
        encoded = self.encoder(image.flatten(0, 1))
        position = agent_pos.flatten(0, 1) * agent_pos.new_tensor([1., -1.])
        # This world vector stays fixed when the image and agent are rotated.
        reference = self.symmetry_gate * position.new_tensor([1., 0.]).expand(batch * steps, 2)
        combined = enn.GeometricTensor(torch.cat((encoded, position, reference), -1), self.obs_type)
        return self.obs_projection(combined).tensor.reshape(batch, -1)
