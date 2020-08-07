"""
For the heatmap, I index into the Q function with Q[state, action]
"""

import argparse
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
from visualization.memory_bptt.analyze_critic_watermaze_1d import (
    create_optimal_qf
)

USE_TIME = False


def create_figure(
        title,
        vector_fields,
        target_poses,
        iteration_number
):
    series_length = len(target_poses)
    width = 5 * series_length
    height = 5
    fig, axes = plt.subplots(
        1, series_length, figsize=(width, height)
    )
    for i, (vf, target_pos) in enumerate(
            zip(vector_fields, target_poses)
    ):
        # `heatmaps` is now a list of heatmaps, such that
        # heatmaps[k] = list_of_list_of_heatmaps[k][i]
        min_pos = max(target_pos - WaterMaze.TARGET_RADIUS,
                      -WaterMaze.BOUNDARY_DIST)
        max_pos = min(target_pos + WaterMaze.TARGET_RADIUS,
                      WaterMaze.BOUNDARY_DIST)

        """
        Plot Estimated & Optimal QF
        """
        ax = axes[i]
        vu.plot_vector_field(fig, ax, vf)
        ax.vlines([min_pos, max_pos], *ax.get_ylim())
        ax.set_xlabel("X_velocity")
        ax.set_ylabel("Y_velocity")
        ax.set_title("{0} QF and dQ/dA. Target Position = {1}".format(
            title,
            target_pos,
        ))

    fig.suptitle("Iteration = {0}".format(iteration_number))
    return fig


def create_qf_derivative_eval_fnct(qf, target_x_pos, time):
    def evaluate(x, y):
        pos = np.array([0, 0])
        target_pos = np.array([target_x_pos, 0])
        dist = np.linalg.norm(pos - target_pos)
        on_target = dist <= WaterMaze.TARGET_RADIUS
        state = np.hstack([pos, on_target, time, target_pos])
        state = Variable(torch.from_numpy(state)).float().unsqueeze(0)

        vel = [x, y]
        action = np.array([vel])
        action_var = Variable(
            torch.from_numpy(action).float(),
            requires_grad=True,
        )
        out = qf(state, action_var)
        out.backward()
        dx1, dx2 = action_var.grad.data.numpy().flatten()
        value = out.data.numpy().flatten()[0]
        return value, dx1, dx2
    return evaluate


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

        vector_fields_t0 = []
        optimal_heatmaps = []
        target_poses = np.linspace(-5, 5, num=5)
        for target_pos in target_poses:
            qf_vector_field_eval = create_qf_derivative_eval_fnct(qf, target_pos, 0)
            vector_fields_t0.append(vu.make_vector_field(
                qf_vector_field_eval,
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
            'Estimated T=0',
            vector_fields_t0,
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
