"""
For the heatmap, I index into the Q function with Q[action_1, action_2].
The state is fixed

Usage:

```
python <this_script>.py path/to/snapshot/dir
```
"""

import argparse
import os
import os.path as osp
import re
import subprocess
from operator import itemgetter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np

import rlkit.visualization.visualization_util as vu
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.multitask.reacher_env import position_from_angles
from rlkit.misc.html_report import HTMLReport
from rllab.misc.instrument import query_yes_no

USE_TIME = True


def create_figure(
        titles,
        heatmaps,
):
    num_heatmaps = len(heatmaps)
    width = 5 * num_heatmaps
    height = 5
    fig, axes = plt.subplots(1, num_heatmaps, figsize=(width, height))
    for i, (title, heatmap) in enumerate(zip(titles, heatmaps)):
        """
        Plot Estimated & Optimal QF
        """
        if num_heatmaps == 1:
            ax = axes
        else:
            ax = axes[i]
        vu.plot_heatmap(heatmap, fig, ax)
        ax.set_xlabel("X-action")
        ax.set_ylabel("Y-action")
        ax.set_title(title)

    return fig


def create_qf_eval_fnct(qf, start_state, goal_state):
    def evaluate(x, y):
        action = np.array([x, y])
        action = ptu.np_to_var(action).unsqueeze(0)
        state = ptu.np_to_var(start_state).unsqueeze(0)
        goal_states = ptu.np_to_var(goal_state).unsqueeze(0)
        discount = ptu.np_to_var(np.array([[0]]))
        out = qf(state, action, goal_states, discount)
        return out.data.numpy()

    return evaluate


def get_path_and_iters(dir_path):
    path_and_iter = []
    for pkl_path in dir_path.glob('*.pkl'):
        if 'data.pkl' in str(pkl_path):
            continue
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
    # parser.add_argument("--num_iters", type=int)
    args = parser.parse_args()
    base = Path(os.getcwd())
    base = base / args.folder_path

    path_and_iter = get_path_and_iters(base)

    resolution = 20
    x_bounds = (-1, 1)
    y_bounds = (-1, 1)

    report = HTMLReport(
        str(base / 'report.html'), images_per_row=1
    )

    # for path, itr in takespread(path_and_iter, args):
    for path, itr in [
        path_and_iter[-1]
    ]:
        report.add_text("Path: %s" % path)
        print("Loading: %s" % path)
        data = joblib.load(str(path))
        qf = data['qf']
        env = data['env']
        qf.train(False)

        start_state = env.reset()
        report.add_text("Start State = {}".format(start_state))
        report.add_text(
            "Start XY = {}".format(
                position_from_angles(np.expand_dims(start_state, 0))
            )
        )
        goal_states = [start_state]
        goal_states += [
            env.sample_goal_for_rollout()
            for _ in range(5)
        ]
        for goal_state in goal_states:
            qf_eval = create_qf_eval_fnct(qf, start_state, goal_state)
            qf_heatmap = vu.make_heat_map(
                qf_eval,
                x_bounds=x_bounds,
                y_bounds=y_bounds,
                resolution=resolution,
            )

            fig = create_figure(
                ['Estimated'],
                [qf_heatmap],
            )
            img = vu.save_image(fig)
            report.add_image(img, "Goal State = {}\nGoal XY = {}".format(
                goal_state,
                position_from_angles(np.expand_dims(goal_state, 0))
            ))

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
