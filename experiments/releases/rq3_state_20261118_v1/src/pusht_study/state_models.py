"""RQ3 state-input baselines and explicit symmetry-breaking interventions."""

import torch
from torch import nn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from .equidiff import PlanarEquiUNet
from escnn import gspaces, nn as enn
from escnn.group import CyclicGroup


class StateDP(nn.Module):
    def __init__(self, config):
        super().__init__()
        tiny = config.get("tiny", False)
        width = 16 if tiny else 256
        self.encoder = nn.Sequential(nn.Linear(6, width), nn.Mish(), nn.Linear(width, width), nn.Mish())
        self.unet = ConditionalUnet1D(input_dim=2, global_cond_dim=config["obs_steps"] * width,
                                      diffusion_step_embed_dim=32 if tiny else 128,
                                      down_dims=(16, 32) if tiny else config["down_dims"],
                                      kernel_size=5, n_groups=8, cond_predict_scale=True)

    def encode(self, state, goal):
        return self.encoder(state).flatten(1)

    def predict_encoded(self, sample, timestep, encoded):
        return self.unet(sample, timestep, global_cond=encoded)

    def forward(self, sample, timestep, state, goal):
        return self.predict_encoded(sample, timestep, self.encode(state, goal))


class StateEqui(nn.Module):
    uses_escnn = True

    def __init__(self, config):
        super().__init__()
        self.method = config["method"]
        tiny = config.get("tiny", False)
        hidden = 8 if tiny else config.get("state_hidden", 64)
        group = gspaces.no_base_space(CyclicGroup(4))
        extra = self.method in ("state_equidiff_goal", "state_equidiff_gate")
        self.in_type = enn.FieldType(group, (3 + int(extra)) * [group.irrep(1)])
        out_type = enn.FieldType(group, hidden * [group.regular_repr])
        self.encoder = enn.SequentialModule(enn.Linear(self.in_type, out_type), enn.ReLU(out_type),
                                            enn.Linear(out_type, out_type), enn.ReLU(out_type))
        if self.method == "state_equidiff_gate":
            self.symmetry_gate = nn.Parameter(torch.zeros(()))
        self.diff = PlanarEquiUNet(act_emb_dim=8 if tiny else 32, local_cond_dim=None,
                                  global_cond_dim=config["obs_steps"] * hidden,
                                  diffusion_step_embed_dim=32 if tiny else 128,
                                  down_dims=(16, 32) if tiny else config["down_dims"],
                                  kernel_size=5, n_groups=8, cond_predict_scale=True, N=4)
        self.train()

    def encode(self, state, goal):
        batch, steps = state.shape[:2]
        vectors = state.reshape(batch, steps, 3, 2)
        if self.method == "state_equidiff_goal":
            vectors = torch.cat((vectors, goal.unsqueeze(-2)), dim=-2)
        elif self.method == "state_equidiff_gate":
            # A fixed world reference is intentionally NOT rotated with the scene.
            reference = self.symmetry_gate * state.new_tensor([1., 0.]).expand(batch, steps, 1, 2)
            vectors = torch.cat((vectors, reference), dim=-2)
        cartesian = vectors * state.new_tensor([1., -1.])
        feature = enn.GeometricTensor(cartesian.reshape(batch * steps, -1), self.in_type)
        return self.encoder(feature).tensor.reshape(batch, -1)

    def predict_encoded(self, sample, timestep, encoded):
        return self.diff(sample, timestep, global_cond=encoded)

    def forward(self, sample, timestep, state, goal):
        return self.predict_encoded(sample, timestep, self.encode(state, goal))


def make_state_model(config):
    if config["method"] == "state_dp":
        return StateDP(config)
    if config["method"] in ("state_equidiff", "state_equidiff_goal", "state_equidiff_gate"):
        return StateEqui(config)
    raise ValueError(config["method"])
