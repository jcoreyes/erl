import torch
import numpy as np
from torch import nn as nn
from torch.autograd import Variable as TorchVariable
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


def kronecker_product(t1, t2):
    """
    Computes the Kronecker product between two tensors
    See https://en.wikipedia.org/wiki/Kronecker_product
    """
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (
        t1.unsqueeze(2)
          .unsqueeze(3)
          .repeat(1, t2_height, t2_width, 1)
          .view(out_height, out_width)
    )

    return expanded_t1 * tiled_t2


def selu(
        x,
        alpha=1.6732632423543772848170429916717,
        scale=1.0507009873554804934193349852946,
):
    """
    Based on https://github.com/dannysdeng/selu/blob/master/selu.py
    """
    return scale * (
        F.relu(x) + alpha * (F.elu(-1 * F.relu(-1 * x)))
    )


def alpha_dropout(
        x,
        p=0.05,
        alpha=-1.7580993408473766,
        fixedPointMean=0,
        fixedPointVar=1,
        training=False,
):
    keep_prob = 1 - p
    if keep_prob == 1 or not training:
        return x
    a = np.sqrt(fixedPointVar / (keep_prob * (
        (1 - keep_prob) * pow(alpha - fixedPointMean, 2) + fixedPointVar)))
    b = fixedPointMean - a * (
        keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
    keep_prob = 1 - p

    random_tensor = keep_prob + torch.rand(x.size())
    binary_tensor = Variable(torch.floor(random_tensor))
    x = x.mul(binary_tensor)
    ret = x + alpha * (1 - binary_tensor)
    ret.mul_(a).add_(b)
    return ret


def alpha_selu(x, training=False):
    return alpha_dropout(selu(x), training=training)


def double_moments(x, y):
    """
    Returns the first two moments between x and y.

    Specifically, for each vector x_i and y_i in x and y, compute their
    outer-product. Flatten this resulting matrix and return it.

    The first moments (i.e. x_i and y_i) are included by appending a `1` to x_i
    and y_i before taking the outer product.
    :param x: Shape [batch_size, feature_x_dim]
    :param y: Shape [batch_size, feature_y_dim]
    :return: Shape [batch_size, (feature_x_dim + 1) * (feature_y_dim + 1)
    """
    batch_size, x_dim = x.size()
    _, y_dim = x.size()
    x = torch.cat((x, ptu.Variable(torch.ones(batch_size, 1))), dim=1)
    y = torch.cat((y, ptu.Variable(torch.ones(batch_size, 1))), dim=1)
    x_dim += 1
    y_dim += 1
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    outer_prod = (
        x.expand(batch_size, x_dim, y_dim) * y.expand(batch_size, x_dim, y_dim)
    )
    return outer_prod.view(batch_size, -1)


"""
GPU wrappers
"""
_use_gpu = False


def set_gpu_mode(mode):
    global _use_gpu
    _use_gpu = mode


def gpu_enabled():
    return _use_gpu


# noinspection PyPep8Naming
def FloatTensor(*args, **kwargs):
    if _use_gpu:
        return torch.cuda.FloatTensor(*args, **kwargs)
    else:
        # noinspection PyArgumentList
        return torch.FloatTensor(*args, **kwargs)


def Variable(*args, **kwargs):
    var = TorchVariable(*args, **kwargs)
    if _use_gpu:
        var = var.cuda()
    return var


def from_numpy(*args, **kwargs):
    if _use_gpu:
        return torch.from_numpy(*args, **kwargs).cuda()
    else:
        return torch.from_numpy(*args, **kwargs)


def get_numpy(tensor):
    if _use_gpu:
        return tensor.data.cpu().numpy()
    return tensor.data.numpy()
