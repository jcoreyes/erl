"""
Plot histogram of the actions, observations, computed rewards, and whatever
else I might find useful in a replay buffer.
"""
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
        env = data['env']

    train_replay_buffer = replay_buffer.train_replay_buffer

    obs = train_replay_buffer._observations[:train_replay_buffer._size, :]
    num_features = obs.shape[-1]
    fig, axes = plt.subplots(num_features)
    for i in range(num_features):
        ax = axes[i]
        diff = obs[:-1, i] - obs[1:, i]
        diff *= (
            1-train_replay_buffer._final_state[:train_replay_buffer._size-1]
        )
        ax.hist(diff)
        ax.set_title("Next obs - obs, dim #{}".format(i+1))
    plt.show()

    actions = train_replay_buffer._actions
    action_dim = actions.shape[-1]
    fig, axes = plt.subplots(action_dim)
    for i in range(action_dim):
        ax = axes[i]
        x = actions[:train_replay_buffer._size, i]
        ax.hist(x)
        ax.set_title("actions, dim #{}".format(i+1))
    plt.show()

    obs = train_replay_buffer._observations
    num_features = obs.shape[-1]
    fig, axes = plt.subplots(num_features)
    for i in range(num_features):
        ax = axes[i]
        x = obs[:train_replay_buffer._size, i]
        ax.hist(x)
        ax.set_title("observations, dim #{}".format(i+1))
    plt.show()

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
    ax.hist(computed_rewards)
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
