from fanova import fANOVA
import numpy as np
from collections import defaultdict, namedtuple

from railrl.misc.data_processing import get_data_and_variants
import ConfigSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

FanovoInfo = namedtuple(
    'FanovaInfo', ['f', 'config_space', 'X', 'Y', 'param_names']
)


def is_numeric(x):
    return isinstance(x, int) or isinstance(x, float)


def get_fanova_info(base_dir):
    data_and_variants = get_data_and_variants(base_dir)
    data, all_variants = zip(*data_and_variants)
    Y = np.array([exp['AverageReturn'][-1] for exp in data])
    variants = filter_variants_to_unique_params(all_variants)
    names = list(variants[0].keys())
    X_raw = _extract_features(variants, names)
    config_space, X = _get_config_space_and_new_features(X_raw, names)
    return FanovoInfo(
        fANOVA(X, Y, config_space=config_space),
        config_space,
        X,
        Y,
        names,
    )


def filter_variants_to_unique_params(all_variants):
    variant_key_to_values = defaultdict(set)
    for variant in all_variants:
        for k, v in variant.items():
            if type(v) == list:
                v = str(v)
            variant_key_to_values[k].add(v)
    filtered_variants = []
    for variant in all_variants:
        new_variant = {
            k: v for k, v in variant.items()
            if len(variant_key_to_values[k]) > 1 and is_numeric(v)
        }
        filtered_variants.append(new_variant)
    return filtered_variants


def _extract_features(all_variants, names):
    num_examples = len(all_variants)
    feature_dim = len(names)
    X = np.zeros((num_examples, feature_dim))
    for ex_i, variant in enumerate(all_variants):
        for feature_i, name in enumerate(names):
            X[ex_i, feature_i] = variant[name]
    return X


def _get_config_space_and_new_features(X, names):
    config_space = ConfigSpace.ConfigurationSpace()
    for name, mn, mx in zip(names, np.min(X, axis=0), np.max(X, axis=0)):
        config_space.add_hyperparameter(UniformFloatHyperparameter(name, mn, mx))

    # Not sure why, but config_space shuffles the order of the hyperparameters
    new_name_order = [
        config_space.get_hyperparameter_by_idx(i) for i in range(len(names))
    ]
    new_order = [names.index(name) for name in new_name_order]
    return config_space, X[:, new_order]
