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
import seaborn as sns
import torch
from torch.autograd import Variable

from railrl.envs.pygame.water_maze import WaterMaze
import railrl.misc.visualization_util as vu
from visualization.memory_bptt.analyze_critic_watermaze_1d import (
    create_qf_eval_fnct, create_optimal_qf
)

USE_TIME = False


def create_figure(
        target_poses,
        *list_of_vector_fields
):
    series_length = len(target_poses)
    num_vfs = len(list_of_vector_fields)
    width = 7 * series_length
    height = 7 * num_vfs
    fig, axes = plt.subplots(
        num_vfs, series_length, figsize=(width, height)
    )
    for i, target_pos in enumerate(target_poses):
        for j, vf in enumerate([vfs[i] for vfs in list_of_vector_fields]):
            # `heatmaps` is now a list of heatmaps, such that
            # heatmaps[k] = list_of_list_of_heatmaps[k][i]
            min_pos = max(target_pos - WaterMaze.TARGET_RADIUS,
                          -WaterMaze.BOUNDARY_DIST)
            max_pos = min(target_pos + WaterMaze.TARGET_RADIUS,
                          WaterMaze.BOUNDARY_DIST)

            """
            Plot Estimated & Optimal QF
            """
            ax = axes[j][i]
            vu.plot_vector_field(fig, ax, vf)
            ax.vlines([min_pos, max_pos], *ax.get_ylim())
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
            ax.set_title("{0}. t = {1}. Target X Pos = {2}".format(
                vf.info['title'],
                vf.info['time'],
                vf.info['target_pos'],
            ))

    return fig


def create_qf_derivative_eval_fnct(qf, target_pos, time):
    def evaluate(pos, vel):
        dist = np.linalg.norm(pos - target_pos)
        on_target = dist <= WaterMaze.TARGET_RADIUS
        state = np.hstack([pos, on_target, time, target_pos])
        state = Variable(torch.from_numpy(state)).float().unsqueeze(0)

        action = np.array([[vel]])
        action_var = Variable(
            torch.from_numpy(action).float(),
            requires_grad=True,
        )
        out = qf(state, action_var)
        out.backward()
        dq_da = action_var.grad.data.numpy().flatten()
        value = out.data.numpy().flatten()[0]
        return value, 0, dq_da
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

        target_poses = np.linspace(-5, 5, num=5)
        list_of_vector_fields = []
        for time in [0, 24]:
            vector_fields = []
            for target_pos in target_poses:
                qf_vector_field_eval = create_qf_derivative_eval_fnct(
                    qf, target_pos, time
                )
                vector_fields.append(vu.make_vector_field(
                    qf_vector_field_eval,
                    x_bounds=state_bounds,
                    y_bounds=action_bounds,
                    resolution=resolution,
                    info=dict(
                        time=time,
                        target_pos=target_pos,
                        title="Estimated QF and dQ/da",
                    )
                ))
            list_of_vector_fields.append(vector_fields)

        fig = create_figure(
            target_poses,
            *list_of_vector_fields,
            # vector_fields_t0,
            # vector_fields_t24,
        )
        fig.suptitle("Iteration = {0}".format(iter_number))
        save_dir = base / "derivative_images"
        if not save_dir.exists():
            save_dir.mkdir()
        save_file = save_dir / 'iter_{}.png'.format(iter_number)
        fig.savefig(str(save_file), bbox_inches='tight')

if __name__ == '__main__':
    main()
