import torch
import railrl.torch.pytorch_util as ptu
import numpy as np

from railrl.data_management.normalizer import Normalizer


class TorchNormalizer(Normalizer):
    """
    Update with np array, but de/normalize pytorch Tensors.
    """
    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean = ptu.np_to_var(self.mean, requires_grad=False)
        std = ptu.np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean = ptu.np_to_var(self.mean, requires_grad=False)
        std = ptu.np_to_var(self.std, requires_grad=False)
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std
