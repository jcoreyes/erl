import torch
from torch import nn
from torch.nn import __init__

from railrl.torch.pytorch_util import activation_from_string


class Clamp(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.__name__ = "Clamp"

    def forward(self, x):
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


class ConcatTuple(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.cat(inputs, dim=self.dim)


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)


class MultiInputSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class Splitter(nn.Module):
    """
           .-> head 0
          /
    input ---> head 1
          \
           '-> head 2
    """
    def __init__(
            self,
            output_sizes,
            output_activations=None,
    ):
        super().__init__()
        if output_activations is None:
            output_activations = ['identity' for _ in output_sizes]
        else:
            if len(output_activations) != len(output_sizes):
                raise ValueError("output_activation and output_sizes must have "
                                 "the same length")

        self._output_narrow_params = []
        self._output_activations = []
        for output_activation in output_activations:
            if isinstance(output_activation, str):
                output_activation = activation_from_string(output_activation)
            self._output_activations.append(output_activation)
        start_idx = 0
        for output_size in output_sizes:
            self._output_narrow_params.append((start_idx, output_size))
            start_idx = start_idx + output_size

    def forward(self, flat_outputs):
        pre_activation_outputs = tuple(
            flat_outputs.narrow(1, start, length)
            for start, length in self._output_narrow_params
        )
        outputs = tuple(
            activation(x)
            for activation, x in zip(
                self._output_activations, pre_activation_outputs
            )
        )
        return outputs