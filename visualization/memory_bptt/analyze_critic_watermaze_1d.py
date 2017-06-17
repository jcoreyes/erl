from collections import namedtuple

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.autograd import Variable

from railrl.envs.pygame.water_maze import WaterMaze

HeatMap = namedtuple("HeatMap", ['values', 'x_range', 'y_range'])

file = (
    '/home/vitchyr/git/rllab-rail/railrl/data/local/6-16-launch-benchmark-1d-ddpg-correct/6-16-launch-benchmark-1d-ddpg-correct_2017_06_16_19_05_45_0001--s-0'
    '/params.pkl'
)

data = joblib.load(file)
policy = data['policy']
qf = data['qf']
env = data['env']
print("Policy loaded")


def make_heat_map(eval_func, *, resolution=50, min_val=-1, max_val=1):
    linspace = np.linspace(min_val, max_val, num=resolution)
    map = np.zeros((resolution, resolution))
    x_range = linspace
    y_range = linspace

    for i in range(resolution):
        for j in range(resolution):
            map[i, j] = eval_func(linspace[i], linspace[j])
    return HeatMap(map, x_range, y_range)


def plot_maps(heatmaps, target_poses):
    fig, axes = plt.subplots(1, len(heatmaps))
    for ax, heatmap, target_pos in zip(axes, heatmaps, target_poses):
        title = "Target_Pos = {}".format(target_pos)
        sns.heatmap(
            heatmap.values,
            ax=ax,
            # xticklabels=heatmap.x_range,
            # yticklabels=heatmap.y_range,
            xticklabels=False,
            yticklabels=False,
        )
        plt.plot((target_pos, target_pos), (-1, 1), 'k-')
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title(title)
    plt.show()


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

heatmaps = []
target_poses = np.linspace(-5, 5, num=5)
for target_pos in target_poses:
    qf_eval = create_eval_fnct(qf, target_pos)
    heatmaps.append(make_heat_map(qf_eval))

plot_maps(heatmaps, target_poses)
