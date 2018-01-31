import torch

from railrl.torch.core import PyTorchModule


class ModelToImplicitModel(PyTorchModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, action, next_obs):
        out = (next_obs - obs - self.model(obs, action))**2
        return -torch.norm(out, dim=1).unsqueeze(1)