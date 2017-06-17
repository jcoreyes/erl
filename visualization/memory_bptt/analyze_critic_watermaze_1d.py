from collections import namedtuple

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable
from pathlib import Path
import re
from operator import itemgetter

from railrl.envs.pygame.water_maze import WaterMaze

HeatMap = namedtuple("HeatMap", ['values', 'x_values', 'y_values'])


def make_heat_map(eval_func, *, resolution=10, min_val=-1, max_val=1):
    linspace = np.linspace(min_val, max_val, num=resolution)
    map = np.zeros((resolution, resolution))
    x_values = linspace
    y_values = linspace

    for i in range(resolution):
        for j in range(resolution):
            map[i, j] = eval_func(linspace[i], linspace[j])
    return HeatMap(map, x_values, y_values)


def create_figure(heatmaps, target_poses, title_base):
    width = 5 * len(heatmaps)
    height = 5
    fig, axes = plt.subplots(2, len(heatmaps), figsize=(width, height))
    for i, (heatmap, target_pos) in enumerate(zip(heatmaps, target_poses)):
        ax = axes[0][i]
        sns.heatmap(
            heatmap.values,
            ax=ax,
            xticklabels=False,
            yticklabels=False,
            # annot=annotation,
        )

        min_pos = max(target_pos - WaterMaze.TARGET_RADIUS,
                      -WaterMaze.BOUNDARY_DIST)
        max_pos = min(target_pos + WaterMaze.TARGET_RADIUS,
                      WaterMaze.BOUNDARY_DIST)
        x_values = heatmap.x_values
        target_right_of = min_pos <= x_values
        target_left_of = x_values <= max_pos
        first_index_on = np.where(target_right_of)[0][0]
        last_index_on = np.where(target_left_of)[0][-1] + 1
        ax.vlines([first_index_on, last_index_on], *ax.get_xlim())

        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        title = "{0}, Target_Pos = {1}".format(title_base, target_pos)
        ax.set_title(title)

        ax_bottom = axes[1][i]
        ax_bottom.plot(x_values, np.max(heatmap.values, axis=1))
        ax_bottom.vlines([min_pos, max_pos], *ax_bottom.get_ylim())

        ax_bottom.set_xlabel("Position")
        ax_bottom.set_ylabel("Max Value")
        ax_bottom.set_title(title)
    return fig


def create_eval_fnct(qf, target_pos):
    def evaluate(x_pos, x_vel):
        dist = np.linalg.norm(x_pos - target_pos)
        on_target = dist <= WaterMaze.TARGET_RADIUS
        state = np.hstack([x_pos, on_target, target_pos])
        state = Variable(torch.from_numpy(state)).float().unsqueeze(0)

        action = np.array([x_vel])
        action = Variable(torch.from_numpy(action)).float().unsqueeze(0)
        out = qf(state, action)
        return out.data.numpy()
    return evaluate


def main():
    base = Path(
        '/home/vitchyr/git/rllab-rail/railrl/data/local/'
        '6-17-launch-benchmark-1d-ddpg/'
        '6-17-launch-benchmark-1d-ddpg_2017_06_17_13_47_14_0001--s-0'
    )
    path_and_iter = []
    for pkl_path in base.glob('*.pkl'):
        match = re.search('_([0-9]*).pkl$', str(pkl_path))
        iter_number = int(match.group(1))
        path_and_iter.append((pkl_path, iter_number))
    path_and_iter = sorted(path_and_iter, key=itemgetter(1))

    for path, iter_number in path_and_iter:
        data = joblib.load(str(path))
        save_file = base / "images" / 'iter_{}.png'.format(iter_number)
        qf = data['qf']
        print("QF loaded from iteration %d" % iter_number)

        heatmaps = []
        target_poses = np.linspace(-5, 5, num=5)
        for target_pos in target_poses:
            qf_eval = create_eval_fnct(qf, target_pos)
            heatmaps.append(make_heat_map(
                qf_eval,
                resolution=10,
                min_val=-5,
                max_val=5,
            ))

        fig = create_figure(heatmaps, target_poses,
                            "Iteration {}".format(iter_number))
        fig.savefig(str(save_file))

if __name__ == '__main__':
    main()
