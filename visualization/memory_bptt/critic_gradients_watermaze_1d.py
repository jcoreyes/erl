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
from rlkit.misc.html_report import HTMLReport

USE_TIME = False


def create_figure(
        report: HTMLReport,
        target_poses,
        *list_of_vector_fields
):
    num_vfs = len(list_of_vector_fields)
    width = 7
    height = 7 * num_vfs
    for i, target_pos in enumerate(target_poses):
        fig, axes = plt.subplots(
            num_vfs, figsize=(width, height)
        )
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
            ax = axes[j]
            vu.plot_vector_field(fig, ax, vf)
            ax.vlines([min_pos, max_pos], *ax.get_ylim())
            ax.set_xlabel("Position")
            ax.set_ylabel("Velocity")
            ax.set_title("{0}. t = {1}. Target X Pos = {2}".format(
                vf.info['title'],
                vf.info['time'],
                vf.info['target_pos'],
            ))
        img = vu.save_image(fig)
        report.add_image(img, "Target Position = {}".format(target_pos))


def create_qf_derivative_eval_fnct(qf, target_pos, time):
    def evaluate(x_pos, x_vel):
        dist = np.linalg.norm(x_pos - target_pos)
        on_target = dist <= WaterMaze.TARGET_RADIUS
        state = np.hstack([x_pos, on_target, time, target_pos])
        state = Variable(torch.from_numpy(state)).float().unsqueeze(0)

        action = np.array([[x_vel]])
        action = Variable(
            torch.from_numpy(action).float(), requires_grad=True,
        )
        out = qf(state, action)
        out.backward()
        dq_da = action.grad.data.numpy()
        return out.data.numpy(), 0, dq_da
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
    base_dir = Path(os.getcwd())
    base_dir = base_dir / args.folder_path

    path_and_iter = get_path_and_iters(base_dir)

    resolution = 20
    state_bounds = (-WaterMaze.BOUNDARY_DIST, WaterMaze.BOUNDARY_DIST)
    action_bounds = (-1, 1)
    num_target_poses = 5
    target_poses = np.linspace(*state_bounds, num_target_poses)

    report = HTMLReport(
        str(base_dir / 'report.html'), images_per_row=num_target_poses
    )

    report.add_header("test_header")
    report.add_text("test")
    for path, iter_number in path_and_iter:
        data = joblib.load(str(path))
        qf = data['qf']
        print("QF loaded from iteration %d" % iter_number)

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

        report.add_text("Iteration = {0}".format(iter_number))
        create_figure(
            report,
            target_poses,
            *list_of_vector_fields,
        )
        report.new_row()

if __name__ == '__main__':
    main()
