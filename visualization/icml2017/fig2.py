"""
Create plot of performance for varying subtrajectory lengths.

Based on data from

/home/vitchyr/git/rllab-rail/railrl/data/papers/icml2017/ablation
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from os.path import join
import json
from collections import defaultdict, OrderedDict
from rlkit.pythonplusplus import nested_dict_to_dot_map_dict
import seaborn

def main():
    # matplotlib.rcParams.update({'font.size': 39})

    all_data = OrderedDict()
    # all_data['Supervised Learning'] = {
    #     'mean': np.array([
    #         1, 1, 1, 1, 1,
    #     ]),
    #     'std': np.array([
    #         0, 0, 0, 0, 0
    #     ]),
    # }
    all_data['Our Method'] = {
        'mean': np.array([
            0.99996094584460005, 0.77999847680324008, 0.83378760993479994,
            0.81999854505065994, 0.99924713224170014, 0.99998145222669987
        ]),
        'std': np.array([
            8.5206995903983193e-05, 0.44508406376284032, 0.32960112187457913,
            0.36619611451006367, 0.0022540571412094082, 7.6906226713638923e-05
        ]),
    }
    all_data['No Memory States Loaded'] = {
        'mean': np.array([
            -0.036391442045599999, 0.45036127958446998, 0.46328121036300002,
             0.28000511065133998, 0.24448918148876003, 0.99999879695484994
        ]),
        'std': np.array([
            0.14982131608715346, 0.71650753363546349, 0.62762382670054939,
             0.66113205179664392, 0.58230533190087608, 3.02435291295703e-06
        ]),
    }
    all_data['No Memory State For Critic'] = {
        'mean': np.array([
            0.32760292261851004, 0.20434100240482606, 0.48140761002900001,
            0.39138742417096994, 0.50663281228384993, 0.49078222304577779
        ]),
        'std': np.array([
            0.44678047236130131, 0.60980865432798537, 0.50864287789616003,
            0.50296096299271631, 0.50042837831944975, 0.51829453495955591
        ]),
    }
    all_data['No Memory States (Truncated BPTT)'] = {
        'mean': np.array([
            0.044998925626400302, 0.18999759227026092, 0.11000021487471001,
            0.28495058849440569, 0.3399999907613, 0.28000548794873065
        ]),
        'std': np.array([
            0.17528572310728113, 0.41032922753233103, 0.34409271916155854,
            0.48425223449102123, 0.55127124703263475, 0.49834466820348694
        ]),
    }

    cmap = matplotlib.cm.get_cmap('plasma')

    rgba = cmap(0)
    print(rgba)

    x_axis = [1, 5, 10, 15, 20, 25]
    N = len(x_axis)
    ind = np.arange(N)
    width = 0.2
    fig, ax = plt.subplots(figsize=(32.0, 20.0))
    r, g, b, a = cmap(0)
    first_color = (r, g, b, a)
    index_to_color_and_pattern = {
        0: (first_color, ''),
        1: (cmap(0.33), '/'),
        2: (cmap(0.66), '.'),
        3: (cmap(1.), 'x'),
    }
    legend_rects = []
    legend_names = []
    for i, (method_name, data) in enumerate(all_data.items()):
        color, pattern = index_to_color_and_pattern[i]
        y_means, y_stds = data['mean'], data['std']
        assert len(y_means) == len(y_stds) == len(x_axis)
        # ax.errorbar(x_axis, y_means, yerr=y_stds)
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
        # rects1 = ax.bar(ind, men_means, width, color='r', yerr=men_std)
        legend_rects.append(rect[0])
        legend_names.append(method_name)
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
