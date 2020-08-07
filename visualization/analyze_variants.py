"""
This script generates a bunch of plots that show

    Final `ylabel` vs variant

Usage:
```
$ python analyze_variants.py path/to/data/folder --ylabel="AverageReturn"
```
"""
import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from rlkit.misc.data_processing import get_trials, \
    get_unique_param_to_values
from rlkit.pythonplusplus import is_numeric


def sort_by_first(*lists):
    combined = zip(*lists)
    sorted_lists = sorted(combined, key=lambda x: x[0])
    return zip(*sorted_lists)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expdir", help="experiment dir, e.g., /tmp/experiments")
    parser.add_argument("--ylabel", default='AverageReturn')
    args = parser.parse_args()
    y_label = args.ylabel

    """
    Load data
    """
    trials = get_trials(args.expdir)

    data = trials[0][0]
    if y_label not in data.dtype.names:
        print("Invalid ylabel. Valid ylabels:")
        for name in sorted(data.dtype.names):
            print(name)
        return

    """
    Get the unique parameters
    """
    _, all_variants = zip(*trials)
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
    fig, axes = plt.subplots(num_params)
    if num_params == 1:
        axes = [axes]
    for i, x_label in enumerate(unique_numeric_param_to_values):
        x_value_to_y_values = defaultdict(list)
        for data, variant in trials:
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
