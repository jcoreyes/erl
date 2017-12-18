import torch
import railrl.torch.pytorch_util as ptu
import numpy as np


class TorchNormalizer(object):
    """
    Update with np array, but de/normalize pytorch Tensors.
    """
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = int(size)
        self.eps = eps * ptu.ones(1)
        self.default_clip_range = default_clip_range
        self.sum = ptu.Variable(ptu.zeros(self.size), requires_grad=False)
        self.sumsq = ptu.Variable(ptu.zeros(self.size), requires_grad=False)
        self.count = ptu.Variable(ptu.ones(1), requires_grad=False)
        self.mean = ptu.Variable(ptu.zeros(self.size), requires_grad=False)
        self.std = ptu.Variable(ptu.ones(self.size), requires_grad=False)
        self.synchronized = True

    def update(self, v):
        assert v.ndim == 2
        assert v.shape[1] == self.size
        self.sum += v.sum(axis=0)
        self.sumsq += (v ** 2).sum(axis=0)
        self.count[0] += v.shape[0]
        self.synchronized = False

    def normalize(self, v, clip_range=None):
        if not self.synchronized:
            self.synchronize()
        if clip_range is None:
            clip_range = self.default_clip_range
        mean, std = self.mean, self.std
        if v.dim() == 2:
            # Unsqueeze along the batch use automatic broadcasting
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return torch.clamp((v - mean) / std, -clip_range, clip_range)

    def denormalize(self, v):
        if not self.synchronized:
            self.synchronize()
        mean, std = self.mean, self.std
        if v.dim() == 2:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return mean + v * std

    def synchronize(self):
        self.sum[...] = torch.mean(self.sum)
        self.sumsq[...] = torch.mean(self.sumsq)
        self.count[...] = torch.mean(self.count)
        self.mean[...] = self.sum / self.count[0]
        self.std[...] = torch.sqrt(
            torch.max(
                self.eps**2,
                self.sumsq / self.count[0] - self.mean ** 2
            )
        )
        self.synchronized = True
