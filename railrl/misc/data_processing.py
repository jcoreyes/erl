"""
Utility functions for writing and loading data.
"""
import json
import numpy as np
import os
import os.path as osp
from collections import OrderedDict, defaultdict, namedtuple
from numbers import Number

from railrl.pythonplusplus import nested_dict_to_dot_map_dict


Trial = namedtuple("Trial", ["data", "variant"])


def matches_dict(criteria_dict, test_dict, ignore_missing_keys=False):
    for k, v in criteria_dict.items():
        if k not in test_dict:
            if ignore_missing_keys:
                return False
            else:
                raise KeyError("Key '{}' not in dictionary".format(k))
        else:
            if test_dict[k] != v:
                return False
    return True


class Experiment(object):
    """
    Represents an experiment, which consists of many Trials.
    """
    def __init__(self, base_dir):
        """
        :param base_dir: A path. Directory structure should be something like:
        ```
        base_dir/
            foo/
                bar/
                    arbtrarily_deep/
                        trial_one/
                            variant.json
                            progress.csv
                        trial_two/
                            variant.json
                            progress.csv
                    trial_three/
                        variant.json
                        progress.csv
                        ...
                    variant.json  # <-- base_dir/foo/bar has its own Trial
                    progress.csv
                variant.json  # <-- base_dir/foo has its own Trial
                progress.csv
            variant.json  # <-- base_dir has its own Trial
            progress.csv
        ```

        The important thing is that `variant.json` and `progress.csv` are
        in the same sub-directory for each Trial.
        """
        self.trials = []
        for data, variant in get_data_and_variants(base_dir):
            self.trials.append(Trial(data, variant))
        assert len(self.trials) > 0, "Nothing loaded."
        self.label = 'AverageReturn'

    def get_trials(self, criteria=None, ignore_missing_keys=False):
        """
        Return a list of Trials that match a criteria.
        :param criteria: A dictionary from key to value that must be matches
        in the trial's variant. e.g.
        ```
        >>> print(exp.trials)
        [
            (X, {'a': True, ...})
            (Y, {'a': False, ...})
            (Z, {'a': True, ...})
        ]
        >>> print(exp.get_trials({'a': True}))
        [
            (X, {'a': True, ...})
            (Z, {'a': True, ...})
        ]
        ```
        :param ignore_missing_keys: If True, ignore a trial if it does not
        have the key provided.
        If False, raise an error.
        :return:
        """
        if criteria is None:
            criteria = {}
        return [trial for trial in self.trials
                if matches_dict(criteria, trial.variant, ignore_missing_keys)]


def create_stats_ordered_dict(
        name,
        data,
        stat_prefix=None,
        always_show_all_stats=False,
        exclude_max_min=False,
):
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

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


def get_dirs(root):
    """
    Get a list of all the directories under this directory.
    """
    yield root
    for root, directories, filenames in os.walk(root):
        for directory in directories:
            yield os.path.join(root, directory)


def get_data_and_variants(base_dir, verbose=False):
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
    delimiter = ','
    for dir_name in get_dirs(base_dir):
        data_file_name = osp.join(dir_name, 'progress.csv')
        # Hack for iclr 2018 deadline
        if not os.path.exists(data_file_name) or os.stat(
                data_file_name).st_size == 0:
            data_file_name = osp.join(dir_name, 'log.txt')
            if not os.path.exists(data_file_name):
                continue
            delimiter = '\t'
        if verbose:
            print("Reading {}".format(data_file_name))
        variant_file_name = osp.join(dir_name, 'variant.json')
        with open(variant_file_name) as variant_file:
            variant = json.load(variant_file)
        variant = nested_dict_to_dot_map_dict(variant)
        num_lines = sum(1 for _ in open(data_file_name))
        if num_lines < 2:
            continue
        data = np.genfromtxt(
            data_file_name, delimiter=delimiter, dtype=None, names=True
        )
        data_and_variants.append((data, variant))
    return data_and_variants


def get_all_csv(base_dir, verbose=False):
    """
    Get a list of all csv data under a directory.
    :param base_dir: root directory
    """
    data = []
    delimiter = ','
    for dir_name in get_dirs(base_dir):
        for data_file_name in os.listdir(dir_name):
            if data_file_name.endswith(".csv"):
                full_path = os.path.join(dir_name, data_file_name)
                if verbose:
                    print("Reading {}".format(full_path))
                data.append(np.genfromtxt(
                    full_path, delimiter=delimiter, dtype=None, names=True
                ))
    return data


def get_unique_param_to_values(all_variants):
    variant_key_to_values = defaultdict(set)
    for variant in all_variants:
        for k, v in variant.items():
            if type(v) == list:
                v = str(v)
            variant_key_to_values[k].add(v)
    unique_key_to_values = {
        k: variant_key_to_values[k]
        for k in variant_key_to_values
        if len(variant_key_to_values[k]) > 1
    }
    return unique_key_to_values