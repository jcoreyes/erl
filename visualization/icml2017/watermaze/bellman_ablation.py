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
import seaborn


def main():
    # matplotlib.rcParams.update({'font.size': 39})
    # base_dir = "/home/vitchyr/git/rllab-rail/rlkit/data/papers/icml2017/watermaze/ablation"
    base_dir = "/home/vitchyr/git/rllab-rail/railrl/data/s3/08-03-generate-bellman-ablation-figure-data/"
    experiment = Experiment(base_dir)

    version_to_list_of_final_scores = OrderedDict()
    subtraj_lengths = [1, 5, 10, 15, 20, 25]
    for write_policy_optimizes,  name in [
        ['both', 'Both'],
        ['bellman', 'Bellman'],
        ['qf', 'Q Function'],
    ]:
        version_to_list_of_final_scores[name] = []
        for subtraj_length in subtraj_lengths:
            trials = experiment.get_trials({
                'algo_params.write_policy_optimizes': write_policy_optimizes,
                'algo_params.subtraj_length': subtraj_length,
            })
            final_scores = np.array([t.data['AverageReturn'][-1] for t in
                                     trials])
            version_to_list_of_final_scores[name].append(final_scores)

    cmap = matplotlib.cm.get_cmap('plasma')

    index_to_color_and_pattern = {
        0: (cmap(0), ''),
        1: (cmap(0.33), '/'),
        2: (cmap(0.66), '.'),
        3: (cmap(1.), 'x'),
    }
    x_axis = subtraj_lengths
    N = len(x_axis)
    ind = np.arange(N)
    width = 0.2
    fig, ax = plt.subplots(figsize=(32.0, 20.0))
    legend_rects = []
    legend_names = []
    for i, (version_name, final_scores) in enumerate(
            version_to_list_of_final_scores.items()
    ):
        color, pattern = index_to_color_and_pattern[i]
        y_means = [np.mean(score) for score in final_scores]
        y_stds = [np.std(score) for score in final_scores]
        assert len(y_means) == len(y_stds) == len(x_axis)
        rect = ax.bar(
            ind + width * i,
            y_means,
            width,
            color=color,
            yerr=y_stds,
            hatch=pattern,
            ecolor='red',
            capsize=10,
            linewidth=10,
        )
        legend_rects.append(rect[0])
        legend_names.append(version_name)
    fontsize = 50
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(x_axis)
    ax.legend(legend_rects, legend_names, prop={'size': 30},
              bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              ncol=2, mode="expand", borderaxespad=0.)
    plt.xlabel("Subtrajectory Length", fontsize=fontsize)
    plt.ylabel("Average Return", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    ltext = plt.gca().get_legend().get_texts()
    plt.setp(ltext[0], fontsize=fontsize)
    plt.savefig("test.png", bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    main()
