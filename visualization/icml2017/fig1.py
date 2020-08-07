"""
Generate plot of Performance vs Iteration for the following method:
    - Our Method
    - DDPG
    - Memory States + DDPG
    - TRPO
    - Memory States + TRPO
"""
import copy
import matplotlib
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
import json
from collections import defaultdict, OrderedDict

from rlkit.misc.data_processing import get_dirs
from rlkit.pythonplusplus import nested_dict_to_dot_map_dict
import seaborn as sns


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expdir",
        help="experiment dir, e.g, /tmp/experiments",
        default="/home/vitchyr/git/rllab-rail/railrl/data/papers/icml2017"
                "/method-vs-mem",
    )
    parser.add_argument("--ylabel", default='AverageReturn')
    args = parser.parse_args()
    y_label = args.ylabel

    """
    Load data
    """
    data_and_variant = []
    dir_names = list(get_dirs(args.expdir))
    print(dir_names)
    method_to_data = OrderedDict()
    for name in [
        'Our Method',
        'DDPG',
        'Memory States + DDPG',
        'TRPO',
        'Memory States + TRPO',
    ]:
        method_to_data[name] = []
    for dir_name in dir_names:
        data_file_name = join(dir_name, 'progress.csv')
        if not os.path.exists(data_file_name):
            continue
        print("Reading {}".format(data_file_name))
        variant_file_name = join(dir_name, 'variant.json')
        with open(variant_file_name) as variant_file:
            variant = json.load(variant_file)
        method_name = variant['version']
        data = np.genfromtxt(data_file_name, delimiter=',', dtype=None, names=True)
        returns = data[y_label]
        method_to_data[method_name].append(returns)

    print(method_to_data.keys())
    fig, ax = plt.subplots(figsize=(32.0, 20.0))
    method_names = []
    cmap = matplotlib.cm.get_cmap('plasma')
    num_values = len(method_to_data)
    index_to_color_and_pattern = {
        i: cmap(i / (num_values - 1)) for i in range(num_values)
    }
    index_to_pattern = {
        0: '-',
        1: '--',
        2: ':',
        3: '-.',
        4: '-',
    }
    for i, (method, data) in enumerate(method_to_data.items()):
        method_names.append(method)
        data_combined = np.vstack(data)
        y_means = np.mean(data_combined, axis=0)
        y_stds = np.std(data_combined, axis=0)
        x_values = np.arange(len(y_means))
        color = index_to_color_and_pattern[i]
        pattern = index_to_pattern[i]
        print(color)
        sns.tsplot(data=data_combined,
                   color=color, linestyle=pattern,
                   condition=method)
        # ax.errorbar(x_values, y_means, yerr=y_stds,
        #     color=color, linestyle=pattern,
        #     condition=method,
        #             )
    fontsize = 50
    ax.set_ylabel("Average Return", fontsize=fontsize)
    ax.set_xlabel("Environment samples (x100)", fontsize=fontsize)
    #ax.legend(method_names, loc='center right')
    ax.legend(method_names, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.,
              markerscale=10)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ltext = plt.gca().get_legend().get_texts()
    plt.setp(ltext[0], fontsize=fontsize)
    plt.savefig("tmp.png", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
