import copy
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import json
from collections import defaultdict
from railrl.pythonplusplus import nested_dict_to_dot_map_dict
import seaborn


def sort_by_first(*lists):
    combined = zip(*lists)
    sorted_lists = sorted(combined, key=lambda x: x[0])
    return zip(*sorted_lists)


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


def is_numeric(x):
    return isinstance(x, int) or isinstance(x, float)


def get_dirs(root):
    for root, directories, filenames in os.walk(root):
        for directory in directories:
            yield os.path.join(root, directory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", help="experiment dir, e.g., /tmp/experiments")
    parser.add_argument("--ylabel", default='AverageReturn')
    args = parser.parse_args()
    y_label = args.ylabel

    """
    Load data
    """
    data_and_variant = []
    dir_names = list(get_dirs(args.expdir))
    print(dir_names)
    for dir_name in dir_names:
        data_file_name = join(dir_name, 'progress.csv')
        if not os.path.exists(data_file_name):
            continue
        print("Reading {}".format(data_file_name))
        variant_file_name = join(dir_name, 'variant.json')
        with open(variant_file_name) as variant_file:
            variant = json.load(variant_file)
        variant = nested_dict_to_dot_map_dict(variant)
        data = np.genfromtxt(data_file_name, delimiter=',', dtype=None, names=True)
        data_and_variant.append((data, variant))

    if y_label not in data.dtype.names:
        print("Invalid ylabel. Valid ylabels:")
        for name in sorted(data.dtype.names):
            print(name)
        return

    import ipdb; ipdb.set_trace()

    """
    Get the unique parameters
    """
    _, all_variants = zip(*data_and_variant)
    unique_param_to_values = get_unique_param_to_values(all_variants)
    unique_numeric_param_to_values = {
        k: unique_param_to_values[k]
        for k in unique_param_to_values
        if is_numeric(list(unique_param_to_values[k])[0])
    }
    # TODO(vitchyr): Use bar plot if xlabel is not numeric, rather than just
    # ignoring it
    value_to_unique_params = defaultdict(dict)

    """
    Plot results
    """
    num_params = len(unique_numeric_param_to_values)
    fig, axes = plt.subplots()
    if num_params == 1:
        axes = [axes]
    for i, x_label in enumerate(unique_numeric_param_to_values):
        x_value_to_y_values = defaultdict(list)
        for data, variant in data_and_variant:
            if len(data[y_label]) > 0:
                print("WARNING. data is missing this label: {}".format(y_label))
                x_value_to_y_values[variant[x_label]].append(data[y_label][-1])
        y_means = []
        y_stds = []
        x_values = []
        for x_value, y_values in x_value_to_y_values.items():
            x_values.append(x_value)
            y_means.append(np.mean(y_values))
            y_stds.append(np.std(y_values))
            value_to_unique_params[np.mean(y_values)][x_label] = x_value

        x_values, y_means, y_stds = sort_by_first(x_values, y_means, y_stds)

        print(x_values)
        print(list(y_means))
        print(list(y_stds))
        axes[i].errorbar(x_values, y_means, yerr=y_stds)
        axes[i].set_ylabel(y_label)
        axes[i].set_xlabel(x_label)

    """
    Display information about the best parameters
    """
    value_and_unique_params = sorted(value_to_unique_params.items(),
                                     key=lambda v_and_params: -v_and_params[0])
    unique_params = list(unique_numeric_param_to_values.keys())
    default_params = {
        k: variant[k]
        for k in variant
        if k not in unique_params
    }
    print("Default Param", default_params)
    print("Top 3 params")
    for value, params in value_and_unique_params[:3]:
        for k, v in params.items():
            print("\t{}: {}".format(k, v))
        print("Value", value)

    plt.show()

if __name__ == '__main__':
    main()
