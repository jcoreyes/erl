import torch
from torch import nn


class Clamp(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.__name__ = "Clamp"

    def __call__(self, x):
        return torch.clamp(x, **self.kwargs)

class Split(nn.Module):
    """
    Split input and process each chunk with a separate module.
    """
    def __init__(self, module1, module2, split_idx):
        super().__init__()
        self.module1 = module1
        self.module2 = module2
        self.split_idx = split_idx

    def forward(self, x):
        in1 = x[:, :self.split_idx]
        out1 = self.module1(in1)

        in2 = x[:, self.split_idx:]
        out2 = self.module2(in2)

        return out1, out2


class FlattenEach(nn.Module):
    def forward(self, inputs):
        return tuple(x.view(x.size(0), -1) for x in inputs)


class FlattenEachParallel(nn.Module):
    def forward(self, *inputs):
        return tuple(x.view(x.size(0), -1) for x in inputs)


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


class Concat(nn.Module):
    def forward(self, inputs):
        return torch.cat(inputs, dim=1)


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            input = module(*input)
        return input
