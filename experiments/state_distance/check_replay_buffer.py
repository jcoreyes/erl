"""
Plot histogram of the actions, observations, computed rewards, and whatever
else I might find useful in a replay buffer.
"""
from itertools import chain
import joblib

import argparse
import pickle

from railrl.envs.multitask.reacher_env import (
    XyMultitaskSimpleStateReacherEnv,
    GoalStateSimpleStateReacherEnv,
)
import matplotlib.pyplot as plt


def main(dataset_path, only_load_buffer=False):
    """
    :param dataset_path: Path to serialized data.
    :param only_load_buffer: If True, then the path is to a pickle'd version
    of a replay buffer. If False, then the path is to joblib'd version of a
    epoch snapshot.
    :return:
    """
    if only_load_buffer:
        env = XyMultitaskSimpleStateReacherEnv()
        with open(dataset_path, 'rb') as handle:
            replay_buffer = pickle.load(handle)
    else:
        data = joblib.load(dataset_path)
        replay_buffer = data['replay_buffer']
        if 'env' in data:
            env = data['env']
        else:
            # Hack for now...
            env = replay_buffer.train_replay_buffer._env

    train_replay_buffer = replay_buffer.train_replay_buffer

    """
    Plot s_{t+1} - s_t
    """
    obs = train_replay_buffer._observations[:train_replay_buffer._size, :]
    num_features = obs.shape[-1]
    if num_features > 8:
        fig, axes = plt.subplots((num_features+1)//2, 2)
        ax_iter = chain(*axes)
    else:
        fig, axes = plt.subplots(num_features)
        ax_iter = chain(axes)
    for i in range(num_features):
        ax = next(ax_iter)
        diff = obs[:-1, i] - obs[1:, i]
        diff = diff[train_replay_buffer._final_state[:train_replay_buffer._size-1] == 0]
        ax.hist(diff, bins=100)
        ax.set_title("Next obs - obs, dim #{}".format(i+1))
    plt.show()

    """
    Plot actions
    """
    actions = train_replay_buffer._actions
    action_dim = actions.shape[-1]
    fig, axes = plt.subplots(action_dim)
    for i in range(action_dim):
        ax = axes[i]
        x = actions[:train_replay_buffer._size, i]
        ax.hist(x, bins=100)
        ax.set_title("actions, dim #{}".format(i+1))
    plt.show()

    """
    Plot observations
    """
    obs = train_replay_buffer._observations
    num_features = obs.shape[-1]
    if num_features > 8:
        fig, axes = plt.subplots((num_features+1)//2, 2)
        ax_iter = chain(*axes)
    else:
        fig, axes = plt.subplots(num_features)
        ax_iter = chain(axes)
    print("(Min, max) obs")
    for i in range(num_features):
        ax = next(ax_iter)
        x = obs[:train_replay_buffer._size, i]
        ax.hist(x, bins=100)
        print((min(x), max(x)), ",")
        ax.set_title("observations, dim #{}".format(i+1))
    plt.show()

    """
    Plot rewards
    """

    batch_size = 100
    batch = train_replay_buffer.random_batch(batch_size)
    sampled_goal_states = env.sample_goal_states(batch_size)
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

    if isinstance(env, GoalStateSimpleStateReacherEnv):
        differences = batch['next_observations'] - sampled_goal_states
        num_features = differences.shape[-1]
        fig, axes = plt.subplots(num_features)
        for i in range(num_features):
            ax = axes[i]
            x = differences[:, i]
            ax.hist(x)
            ax.set_title("next_obs - goal state, dim #{}".format(i+1))
        plt.show()
    import ipdb; ipdb.set_trace()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_pkl_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--buffer', action='store_true')
    args = parser.parse_args()

    dataset_path = args.replay_pkl_path
    main(dataset_path, args.buffer)
