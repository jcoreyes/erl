"""
Plot histogram of the actions, observations, computed rewards, and whatever
else I might find useful in a replay buffer.
"""
import numpy as np
from itertools import chain
import joblib

import argparse

from rlkit.data_management.her_replay_buffer import HerReplayBuffer
import matplotlib.pyplot as plt


def main(dataset_path, pause=False):
    """
    :param dataset_path: Path to serialized data.
    :return:
    """
    data = joblib.load(dataset_path)
    replay_buffer = data['replay_buffer']
    env = data['env']

    train_replay_buffer = replay_buffer
    size = train_replay_buffer._size
    obs = train_replay_buffer._observations[:size, :]
    obs_delta = obs[1:size, :] - obs[:size - 1, :]
    actions = train_replay_buffer._actions[:size]
    # Actions corresponding to final_state=True are nans
    actions = actions[~np.isnan(actions).any(axis=1)]

    if pause:
        import ipdb; ipdb.set_trace()

    obs_dim = obs.shape[-1]
    action_dim = actions.shape[-1]
    """
    Print general statistics
    """
    print("(Min, max, mean, std) obs")
    for i in range(obs_dim):
        o = obs[:train_replay_buffer._size, i]
        print(
            "Dimension {}".format(i),
            (min(o), max(o), np.mean(o), np.std(o))
        )

    print("")
    print("(Min, max, mean, std) delta obs")
    for i in range(obs_dim):
        delta = obs_delta[:, i]
        print(
            "Dimension {}".format(i),
            (min(delta), max(delta), np.mean(delta), np.std(delta))
        )
    print("")
    print("(Min, max, mean, std) action")
    for i in range(action_dim):
        a = obs[:train_replay_buffer._size, i]
        print(
            "Dimension {}".format(i),
            (min(a), max(a), np.mean(a), np.std(a))
        )

    """
    Print everything again but in transpose
    """
    print("")
    print("")
    print("")
    print("obs", list(range(obs_dim)))
    print("mean", repr(np.mean(obs, axis=0)))
    print("std", repr(np.std(obs, axis=0)))
    print("min", repr(np.min(obs, axis=0)))
    print("max", repr(np.max(obs, axis=0)))
    print("")
    print("delta obs", list(range(obs_dim)))
    print("mean", repr(np.mean(obs_delta, axis=0)))
    print("std", repr(np.std(obs_delta, axis=0)))
    print("min", repr(np.min(obs_delta, axis=0)))
    print("max", repr(np.max(obs_delta, axis=0)))
    print("")
    print("actions", list(range(action_dim)))
    print("mean", repr(np.mean(actions, axis=0)))
    print("std", repr(np.std(actions, axis=0)))
    print("min", repr(np.min(actions, axis=0)))
    print("max", repr(np.max(actions, axis=0)))

    print("# data points = {}".format(size))

    """
    Plot s_{t+1} - s_t
    """
    size = train_replay_buffer._size
    if obs_dim == 1:
        fig = plt.figure()
        ax_iter = chain([plt.gca()])
    elif obs_dim > 8:
        fig, axes = plt.subplots((obs_dim + 1) // 2, 2)
        ax_iter = chain(*axes)
    else:
        fig, axes = plt.subplots(obs_dim)
        ax_iter = chain(axes)
    for i in range(obs_dim):
        ax = next(ax_iter)
        diff = train_replay_buffer._observations[:size,
               i] - train_replay_buffer._next_obs[:size, i]

        ax.hist(diff, bins=100)
        ax.set_title("Next obs - obs, dim #{}".format(i + 1))
    plt.show()

    """
    Plot actions
    """
    if obs_dim == 1:
        fig = plt.figure()
        axes = [plt.gca()]
    else:
        fig, axes = plt.subplots(action_dim)
    for i in range(action_dim):
        ax = axes[i]
        x = actions[:size, i]
        ax.hist(x, bins=100)
        ax.set_title("actions, dim #{}".format(i + 1))
    plt.show()

    """
    Plot observations
    """
    if obs_dim == 1:
        fig = plt.figure()
        ax_iter = chain([plt.gca()])
    elif obs_dim > 8:
        fig, axes = plt.subplots((obs_dim + 1) // 2, 2)
        ax_iter = chain(*axes)
    else:
        fig, axes = plt.subplots(obs_dim)
        ax_iter = chain(axes)
    for i in range(obs_dim):
        ax = next(ax_iter)
        x = obs[:size, i]
        ax.hist(x, bins=100)
        ax.set_title("observations, dim #{}".format(i + 1))
    plt.show()

    """
    Plot rewards
    """

    batch_size = 100
    batch = train_replay_buffer.random_batch(batch_size)
    sampled_goal_states = env.sample_goals(batch_size)
    computed_rewards = env.compute_rewards(
        batch['observations'],
        batch['actions'],
        batch['next_observations'],
        sampled_goal_states
    )
    fig, ax = plt.subplots(1)
    ax.hist(computed_rewards, bins=100)
    ax.set_title("computed rewards")
    plt.show()

    """
    If there's goal states, plot them
    """
    if isinstance(train_replay_buffer, HerReplayBuffer):
        goal_dim = env.goal_dim
        goals = train_replay_buffer._goals[:size, :]
        if goal_dim == 1:
            fig = plt.figure()
            ax_iter = chain([plt.gca()])
        elif goal_dim > 8:
            fig, axes = plt.subplots((goal_dim + 1) // 2, 2)
            ax_iter = chain(*axes)
        else:
            fig, axes = plt.subplots(goal_dim)
            ax_iter = chain(axes)
        for i in range(obs_dim):
            ax = next(ax_iter)
            x = goals[:, i]
            ax.hist(x, bins=100)
            ax.set_title("ogals, dim #{}".format(i + 1))
        plt.show()

    if pause:
        import ipdb; ipdb.set_trace()
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_pkl_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    dataset_path = args.replay_pkl_path
    main(dataset_path, args.pause)
