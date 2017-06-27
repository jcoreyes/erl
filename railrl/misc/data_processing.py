"""
Utility functions for loading data.
"""
from collections import OrderedDict
from numbers import Number

import numpy as np
import os


def create_stats_ordered_dict(name, data, stat_prefix=None):
    if stat_prefix is not None:
        name = "{} {}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, np.ndarray) and data.size == 1:
        return OrderedDict({name: float(data)})

    return OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
        (name + ' Max', np.max(data)),
        (name + ' Min', np.min(data)),
    ])


def get_dirs(root):
    """
    Get a list of all the directories under this directory.
    """
    for root, directories, filenames in os.walk(root):
        for directory in directories:
            yield os.path.join(root, directory)
