"""
Common utility functions for pytorch algorithms.
"""

import numpy as np
from torch.autograd import Variable

from railrl.torch import pytorch_util as ptu


def elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return Variable(ptu.from_numpy(elem_or_tuple).float(), requires_grad=False)


def filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    torch_batch = {
        k: elem_or_tuple_to_variable(x)
        for k, x in filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }
    if len(torch_batch['rewards'].size()) == 1:
        torch_batch['rewards'] = torch_batch['rewards'].unsqueeze(-1)
    if len(torch_batch['terminals'].size()) == 1:
        torch_batch['terminals'] = torch_batch['terminals'].unsqueeze(-1)
    return torch_batch
