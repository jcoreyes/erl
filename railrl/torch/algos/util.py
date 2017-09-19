"""
Common utility functions for pytorch algorithms.
"""
import numpy as np
from collections import OrderedDict

from torch.autograd import Variable

from railrl.torch import pytorch_util as ptu
from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import special


def get_statistics_from_pytorch_dict(
        pytorch_dict,
        mean_stat_names,
        full_stat_names,
        stat_prefix,
):
    """
    :param pytorch_dict: Dictionary, from string to pytorch Tensor
    :param mean_stat_names: List of strings. Add the mean of these
    Tensors to the output
    :param full_stat_names: List of strings. Add all statistics of these
    Tensors to the output
    :param stat_prefix: Prefix to all statistics in outputted dict.
    :return: OrderedDict of statistics
    """
    statistics = OrderedDict()
    for name in mean_stat_names:
        tensor = pytorch_dict[name]
        statistics_name = "{} {} Mean".format(stat_prefix, name)
        statistics[statistics_name] = np.mean(ptu.get_numpy(tensor))

    for name in full_stat_names:
        tensor = pytorch_dict[name]
        data = ptu.get_numpy(tensor)
        statistics.update(create_stats_ordered_dict(
            '{} {}'.format(stat_prefix, name),
            data,
        ))
    return statistics


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


def get_generic_path_information(paths, discount, stat_prefix):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    discounted_returns = [
        special.discount_return(path["rewards"], discount)
        for path in paths
    ]
    rewards = np.hstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                discounted_returns,
                                                stat_prefix=stat_prefix))
    actions = np.vstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))

    return statistics