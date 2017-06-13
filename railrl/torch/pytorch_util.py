import torch
import numpy as np
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def soft_update_from_to(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def maximum_2d(t1, t2):
    # noinspection PyArgumentList
    return torch.max(
        torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2),
        dim=2,
    )[0].squeeze(2)


def kronecker_square(t1, size1, t2, size2):
    """
    Computes the Kronecker product between two square tensors
    See https://en.wikipedia.org/wiki/Kronecker_product

    :param t1:
    :param size1:
    :param t2:
    :param size2:
    :return:
    """
    output_size = size1 * size2
    expanded_t1 = F.upsample_nearest(
        t1.view(1, 1, size1, size1), size2, size2
    ).view(output_size, output_size)

    tiled_t2 = t2.repeat(size1, size1)

    return expanded_t1 * tiled_t2


"""
GPU wrappers
"""
_use_gpu = False


def set_gpu_mode(mode):
    global _use_gpu
    _use_gpu = mode


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.FloatTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.FloatTensor(*args, **kwargs)


def from_numpy(*args, **kwargs):
    if _use_gpu:
        return torch.from_numpy(*args, **kwargs).cuda()
    else:
        return torch.from_numpy(*args, **kwargs)


def get_numpy(tensor):
    if _use_gpu:
        return tensor.data.cpu().numpy()
    return tensor.data.numpy()
