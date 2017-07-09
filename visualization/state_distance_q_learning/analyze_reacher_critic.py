"""
For the heatmap, I index into the Q function with Q[action_1, action_2].
The state is fixed
"""

from math import pi
import argparse
import os
import subprocess
import re
from operator import itemgetter
from pathlib import Path
import os.path as osp

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from gym.envs.mujoco import ReacherEnv
from torch.autograd import Variable

import railrl.misc.visualization_util as vu
from railrl.misc.html_report import HTMLReport
from rllab.misc.instrument import query_yes_no

USE_TIME = True


def create_figure(
        titles,
        heatmaps,
        # target_poses,
        # iteration_number
):
    num_heatmaps = len(heatmaps)
    width = 5 * num_heatmaps
    height = 5
    fig, axes = plt.subplots(1, num_heatmaps, figsize=(width, height))
    for i, (title, heatmap) in enumerate(zip(titles, heatmaps)):
        """
        Plot Estimated & Optimal QF
        """
        ax = axes[i]
        vu.plot_heatmap(fig, ax, heatmap)
        ax.set_xlabel("X-action")
        ax.set_ylabel("Y-action")
        ax.set_title(title)

    return fig


env = ReacherEnv()


def create_qf_eval_fnct(qf, target_pos, joint_angles):
    def evaluate(x, y):
        set_state(target_pos, joint_angles)
        obs = env._get_obs()

        action = np.array([x, y])
        action = Variable(torch.from_numpy(action)).float().unsqueeze(0)
        new_obs = np.hstack((obs, target_pos))
        state = Variable(torch.from_numpy(new_obs)).float().unsqueeze(0)
        out = qf(state, action)
        return out.data.numpy()

    return evaluate


def set_state(target_pos, joint_angles):
    qpos, qvel = np.concatenate([joint_angles, target_pos]), np.zeros(4)
    env.set_state(qpos, qvel)


def create_optimal_qf(target_pos, joint_angles):
    """
    Return
        Q(s, a, s_g, 0) = - ||f(s, a) - s_g||^2
    """
    def qfunction(a1, a2):
        action = np.array([a1, a2])
        set_state(target_pos, joint_angles)
        env.do_simulation(action, env.frame_skip)
        pos = env.get_body_com('fingertip')[:2]
        return -np.linalg.norm(pos - target_pos)

    return qfunction


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
    path, _ = path_and_iter[0]
    data = joblib.load(str(path))
    qf = data['qf']

    resolution = 20
    joint_angles = np.array([pi / 2, pi / 2])
    x_bounds = (-1, 1)
    y_bounds = (-1, 1)

    report = HTMLReport(
        str(base / 'report.html'), images_per_row=1
    )
    report.add_text("Joint Angles = {}".format(joint_angles))

    for target_pos in [
        np.array([0, 0]),
        np.array([.1, .1]),
        np.array([.1, -.1]),
        np.array([-.1, .1]),
        np.array([-.1, -.1]),
    ]:
        qf_eval = create_qf_eval_fnct(qf, target_pos, joint_angles)
        qf_heatmap = vu.make_heat_map(
            qf_eval,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            resolution=resolution,
        )
        optimal_qf_eval = create_optimal_qf(target_pos, joint_angles)
        optimal_heatmap = vu.make_heat_map(
            optimal_qf_eval,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            resolution=resolution,
        )

        fig = create_figure(
            ['Estimated', 'Optimal'],
            [qf_heatmap, optimal_heatmap],
        )
        img = vu.save_image(fig)
        report.add_image(img, "Target Position = {}".format(target_pos))

    abs_path = osp.abspath(report.path)
    print("Report saved to: {}".format(abs_path))
    report.save()
    open_report = query_yes_no("Open report?", default="yes")
    if open_report:
        cmd = "xdg-open {}".format(abs_path)
        print(cmd)
        subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()
