"""
For the heatmap, I index into the Q function with Q[state, action]
"""

import argparse
import itertools
import os
import re
from operator import itemgetter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from rlkit.envs.pygame.water_maze import WaterMaze
import rlkit.visualization.visualization_util as vu

USE_TIME = True


def create_figure(
        titles,
        list_of_list_of_heatmaps,
        target_poses,
        iteration_number
):
    num_heatmaps = len(list_of_list_of_heatmaps)
    series_length = len(target_poses)
    width = 5 * series_length
    height = 10 * num_heatmaps
    fig, axes = plt.subplots(
        2 * num_heatmaps, series_length, figsize=(width, height)
    )
    for i, (heatmaps, target_pos) in enumerate(
            zip(zip(*list_of_list_of_heatmaps), target_poses)
    ):
        # `heatmaps` is now a list of heatmaps, such that
        # heatmaps[k] = list_of_list_of_heatmaps[k][i]
        min_pos = max(target_pos - WaterMaze.TARGET_RADIUS,
                      -WaterMaze.BOUNDARY_DIST)
        max_pos = min(target_pos + WaterMaze.TARGET_RADIUS,
                      WaterMaze.BOUNDARY_DIST)
        state_values = heatmaps[0].x_values

        names_and_heatmaps = list(zip(titles, heatmaps))
        num_heatmaps = len(names_and_heatmaps)

        """
        Plot Estimated & Optimal QF
        """
        for j, (title, heatmap) in enumerate(names_and_heatmaps):
            ax = axes[j][i]
            vu.plot_heatmap(heatmap, fig, ax)
            ax.vlines([min_pos, max_pos], *ax.get_ylim())
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
            ax.set_title("{0} QF. Target Position = {1}".format(
                title,
                target_pos,
            ))

        """
        Plot Estimated & Optimal VF
        """
        for j, (title, heatmap) in enumerate(names_and_heatmaps):
            ax = axes[num_heatmaps + j][i]
            ax.plot(state_values, np.max(heatmap.values, axis=1))
            ax.vlines([min_pos, max_pos], *ax.get_ylim())
            ax.set_xlabel("Position")
            ax.set_ylabel("Value Function")
            ax.set_title("{0} VF. Target Position = {1}".format(
                title,
                target_pos,
            ))

    fig.suptitle("Iteration = {0}".format(iteration_number))
    return fig


def create_qf_eval_fnct(qf, target_pos, time):
    def evaluate(x_pos, x_vel):
        dist = np.linalg.norm(x_pos - target_pos)
        on_target = dist <= WaterMaze.TARGET_RADIUS
        if USE_TIME:
            state = np.hstack([x_pos, on_target, time, target_pos])
        else:
            state = np.hstack([x_pos, on_target, target_pos])
        state = Variable(torch.from_numpy(state)).float().unsqueeze(0)

        action = np.array([x_vel])
        action = Variable(torch.from_numpy(action)).float().unsqueeze(0)
        out = qf(state, action)
        return out.data.numpy()
    return evaluate


def clip(value, min_value, max_value):
    return max(min(value, max_value), min_value)


def find_nearest_idx(array, value):
    return (np.abs(array - value)).argmin()


def create_optimal_qf(target_pos, state_bounds, action_bounds, discount_factor,
                      *, resolution=10):
    """
    Do Q-learning to find the optimal Q-values
    :param target_pos:
    :param state_bounds:
    :param action_bounds:
    :param resolution:
    :return:
    """
    def get_reward(state):
        return int(target_pos - WaterMaze.TARGET_RADIUS
                   <= state
                   <= target_pos + WaterMaze.TARGET_RADIUS)

    qf = np.zeros((resolution, resolution))  # state, action
    state_values = np.linspace(*state_bounds, num=resolution)
    action_values = np.linspace(*action_bounds, num=resolution)
    alpha = 0.1
    for _ in range(1000):
        vf = np.max(qf, axis=1)
        for action_i, state_i in itertools.product(range(resolution),
                                                   range(resolution)):
            next_state = clip(
                action_values[action_i] + state_values[state_i],
                *state_bounds
            )
            next_state_i = find_nearest_idx(state_values, next_state)
            reward = get_reward(state_values[next_state_i])
            qf[state_i, action_i] = (
                (1 - alpha) * qf[state_i, action_i]
                + alpha * (
                    reward + discount_factor * vf[next_state_i]
                )
            )

    def qf_fnct(state, action):
        state_i = find_nearest_idx(state_values, state)
        action_i = find_nearest_idx(action_values, action)
        return qf[state_i, action_i]

    return qf_fnct


def get_path_and_iters(dir_path):
    path_and_iter = []
    for pkl_path in dir_path.glob('*.pkl'):
        match = re.search('_(-*[0-9]*).pkl$', str(pkl_path))
        if match is None:  # only saved one param
            path_and_iter.append((pkl_path, 0))
            break
        iter_number = int(match.group(1))
        path_and_iter.append((pkl_path, iter_number))
    return sorted(path_and_iter, key=itemgetter(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=str)
    args = parser.parse_args()
    base = Path(os.getcwd())
    base = base / args.folder_path

    path_and_iter = get_path_and_iters(base)

    resolution = 10
    discount_factor = 0.99
    state_bounds = (-WaterMaze.BOUNDARY_DIST, WaterMaze.BOUNDARY_DIST)
    action_bounds = (-1, 1)

    for path, iter_number in path_and_iter:
        data = joblib.load(str(path))
        qf = data['qf']
        print("QF loaded from iteration %d" % iter_number)

        heatmaps_t0 = []
        heatmaps_t24 = []
        optimal_heatmaps = []
        target_poses = np.linspace(-5, 5, num=5)
        for target_pos in target_poses:
            qf_eval_t0 = create_qf_eval_fnct(qf, target_pos, 0)
            heatmaps_t0.append(vu.make_heat_map(
                qf_eval_t0,
                x_bounds=state_bounds,
                y_bounds=action_bounds,
                resolution=resolution,
            ))
            qf_eval_t24 = create_qf_eval_fnct(qf, target_pos, 24)
            heatmaps_t24.append(vu.make_heat_map(
                qf_eval_t24,
                x_bounds=state_bounds,
                y_bounds=action_bounds,
                resolution=resolution,
            ))
            optimal_qf_eval = create_optimal_qf(
                target_pos,
                state_bounds=state_bounds,
                action_bounds=action_bounds,
                resolution=resolution,
                discount_factor=discount_factor,
            )
            optimal_heatmaps.append(vu.make_heat_map(
                optimal_qf_eval,
                x_bounds=state_bounds,
                y_bounds=action_bounds,
                resolution=resolution,
            ))

        fig = create_figure(
            ['Estimated T=0', 'Estimated T=24', 'Optimal'],
            [heatmaps_t0, heatmaps_t24, optimal_heatmaps],
            target_poses,
            iter_number,
        )
        save_dir = base / "images"
        if not save_dir.exists():
            save_dir.mkdir()
        save_file = save_dir / 'iter_{}.png'.format(iter_number)
        fig.savefig(str(save_file), bbox_inches='tight')

if __name__ == '__main__':
    main()
