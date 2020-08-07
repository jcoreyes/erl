"""
Ablative analysis for varying subtrajectory lengths.

Based on data from

/home/vitchyr/git/rllab-rail/railrl/data/papers/icml2017/ablation
7-14-bptt-ddpg-watermaze-memory-ablation-memory-state
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from collections import OrderedDict

from rlkit.misc.data_processing import Experiment
import seaborn as sns


def main():
    # fontsize = 50
    fontsize = 12
    linewidth = 5
    plt.rc('legend', fontsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)
    # sns.set_style("whitegrid")
    base_dir = "/home/vitchyr/git/rllab-rail/railrl/data/papers/icml2017" \
               "/watermaze/watermaze-memory"
    experiment = Experiment(base_dir)
    fig, axes = plt.subplots(3, 1, figsize=(10.0, 7.5))

    method_to_our_data = OrderedDict()
    subtraj_lengths = [1, 10, 20, 25]
    for subtraj_length in subtraj_lengths:
        name = "Our Method, Subtrajectory Length = {}".format(subtraj_length)
        trials = experiment.get_trials(
            {
                'algo_params.subtraj_length': subtraj_length,
                'version': "Our Method",
            },
            ignore_missing_keys=True,
        )
        final_scores = [
            t.data['AverageReturn'][:50] for t in trials
            if len(t.data['AverageReturn']) >= 50  # some things crashed
        ]
        method_to_our_data[name] = final_scores

    ax = axes[0]
    method_names = []
    cmap = matplotlib.cm.get_cmap('plasma')
    num_values = len(method_to_our_data)
    index_to_color = {
        i: cmap((i) / (num_values)) for i in range(num_values)
    }
    index_to_linestyle = {
        0: '-',
        1: '--',
        2: ':',
        3: '-.',
    }
    for i, (method, data) in enumerate(method_to_our_data.items()):
        method_names.append(method)
        data_combined = np.vstack(data)
        sns.tsplot(
            data=data_combined,
            color=index_to_color[i],
            linestyle=index_to_linestyle[i],
            condition=method,
            ax=ax,
        )
    ax.set_ylabel("Average Return", fontsize=fontsize)
    ax.set_xlabel("Environment samples (x100)", fontsize=fontsize)
    ax.legend(method_names, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.,
              markerscale=10)


    method_to_ddpg_data = OrderedDict()
    for name in [
        ['DDPG'],
        ['Memory States + DDPG'],
        ['Recurrent DPG'],
        # ['TRPO'],
        # ['Memory States + TRPO'],
        # ['Recurrent TRPO'],
        # ['Our Method'],
    ]:
        name = name[0]
        trials = experiment.get_trials(
            {
                'algo_params.subtraj_length': 25,
                'version': name,
            },
            ignore_missing_keys=True,
        )
        final_scores = [
            t.data['AverageReturn'][:50] for t in trials
            if len(t.data['AverageReturn']) >= 50  # some things crashed
        ]
        method_to_ddpg_data[name] = final_scores

    method_names = []
    cmap = matplotlib.cm.get_cmap('plasma')
    num_values = len(method_to_ddpg_data)
    index_to_color = {
        i: cmap((i) / (num_values)) for i in range(num_values)
    }
    index_to_linestyle = {
        0: '-',
        1: '--',
        2: ':',
        3: '-.',
    }
    ax = axes[1]
    for i, (method, data) in enumerate(method_to_ddpg_data.items()):
        method_names.append(method)
        data_combined = np.vstack(data)
        sns.tsplot(
            data=data_combined,
            color=index_to_color[i],
            linestyle=index_to_linestyle[i],
            condition=method,
            ax=ax,
        )
    ax.set_ylabel("Average Return", fontsize=fontsize)
    ax.set_xlabel("Environment samples (x100)", fontsize=fontsize)
    ax.legend(method_names, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.,
              markerscale=100)


    method_to_trpo_data = OrderedDict()
    for name in [
        ['TRPO'],
        ['Memory States + TRPO'],
        ['Recurrent TRPO'],
    ]:
        name = name[0]
        trials = experiment.get_trials(
            {
                'algo_params.subtraj_length': 25,
                'version': name,
            },
            ignore_missing_keys=True,
        )
        final_scores = [
            t.data['AverageReturn'][:100] for t in trials
            if len(t.data['AverageReturn']) >= 100  # some things crashed
        ]
        method_to_trpo_data[name] = final_scores

    ax = axes[2]
    method_names = []
    cmap = matplotlib.cm.get_cmap('plasma')
    num_values = len(method_to_trpo_data)
    index_to_color = {
        i: cmap((i) / (num_values)) for i in range(num_values)
    }
    index_to_linestyle = {
        0: '-',
        1: '--',
        2: ':',
        3: '-.',
    }
    for i, (method, data) in enumerate(method_to_trpo_data.items()):
        method_names.append(method)
        data_combined = np.vstack(data)
        sns.tsplot(
            data=data_combined,
            color=index_to_color[i],
            linestyle=index_to_linestyle[i],
            condition=method,
            ax=ax,
        )
    ax.set_ylabel("Average Return", fontsize=fontsize)
    ax.set_xlabel("Environment samples (x1000)", fontsize=fontsize)
    ax.legend(method_names, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.,
              markerscale=10)
    # for legobj in legend.legendHandles:
    #     legobj.set_linewidth(linewidth)
    fig.subplots_adjust(hspace=1)
    plt.savefig("comparison.png", bbox_inches='tight', dpi=1000)
    plt.savefig("comparison.eps", bbox_inches='tight')
    plt.savefig("comparison.svg", bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
