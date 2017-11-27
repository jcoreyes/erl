"""
Common evaluation functions for pytorch algorithms.
"""

from collections import OrderedDict

import numpy as np

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch import pytorch_util as ptu
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


def get_difference_statistics(
        statistics,
        stat_names,
        include_validation_train_gap=True,
        include_test_validation_gap=True,
):
    assert include_validation_train_gap or include_test_validation_gap
    difference_pairs = []
    if include_validation_train_gap:
        difference_pairs.append(('Validation', 'Train'))
    if include_test_validation_gap:
        difference_pairs.append(('Test', 'Validation'))
    differences = OrderedDict()
    for prefix_1, prefix_2 in difference_pairs:
        for stat_name in stat_names:
            diff_name = "{0}: {1} - {2}".format(
                stat_name,
                prefix_1,
                prefix_2,
            )
            differences[diff_name] = (
                statistics["{0} {1}".format(prefix_1, stat_name)]
                - statistics["{0} {1}".format(prefix_2, stat_name)]
            )
    return differences


def get_generic_path_information(paths, discount, stat_prefix,
                                 include_discounted_returns=False):
    """
    Get an OrderedDict with a bunch of statistic names and values.
    """
    statistics = OrderedDict()
    returns = [sum(path["rewards"]) for path in paths]

    rewards = np.vstack([path["rewards"] for path in paths])
    statistics.update(create_stats_ordered_dict('Rewards', rewards,
                                                stat_prefix=stat_prefix))
    statistics.update(create_stats_ordered_dict('Returns', returns,
                                                stat_prefix=stat_prefix))
    if include_discounted_returns:
        discounted_returns = [
            special.discount_return(path["rewards"][:, 0], discount)
            for path in paths
        ]
        statistics.update(create_stats_ordered_dict(
            'DiscountedReturns', discounted_returns, stat_prefix=stat_prefix
        ))
    actions = np.hstack([path["actions"] for path in paths])
    statistics.update(create_stats_ordered_dict(
        'Actions', actions, stat_prefix=stat_prefix
    ))
    statistics['Num Paths'] = len(paths)

    return statistics