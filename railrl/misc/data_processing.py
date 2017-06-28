"""
Utility functions for writing and loading data.
"""
import json
import numpy as np
import os
import os.path as osp
from collections import OrderedDict
from numbers import Number

from railrl.pythonplusplus import nested_dict_to_dot_map_dict


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


def get_data_and_variants(base_dir):
    """
    Get a list of (data, variant) tuples, loaded from
        - process.csv
        - variant.json
    files under this directory.
    :param base_dir: root directory
    :return: List of tuples. Each tuple has:
        1. Progress data (nd.array)
        2. Variant dictionary
    """
    data_and_variants = []
    for dir_name in get_dirs(base_dir):
        data_file_name = osp.join(dir_name, 'progress.csv')
        if not os.path.exists(data_file_name):
            continue
        print("Reading {}".format(data_file_name))
        variant_file_name = osp.join(dir_name, 'variant.json')
        with open(variant_file_name) as variant_file:
            variant = json.load(variant_file)
        variant = nested_dict_to_dot_map_dict(variant)
        data = np.genfromtxt(
            data_file_name, delimiter=',', dtype=None, names=True
        )
        data_and_variants.append((data, variant))
    return data_and_variants
