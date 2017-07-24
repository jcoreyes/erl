"""
Plot histogram of the actions and observations in a replay buffer.
"""
import numpy as np
import argparse
import pickle

from railrl.envs.multitask.reacher_env import SimpleReacherEnv
import matplotlib.pyplot as plt


def main(dataset_path):
    with open(dataset_path, 'rb') as handle:
        pool = pickle.load(handle)

    train_pool = pool.train_replay_buffer
    actions = train_pool._actions
    action_dim = actions.shape[-1]
    fig, axes = plt.subplots(action_dim)
    for i in range(action_dim):
        ax = axes[i]
        x = actions[:train_pool._size, i]
        ax.hist(x)
    plt.title("actions")
    plt.show()

    obs = train_pool._observations
    num_features = obs.shape[-1]
    fig, axes = plt.subplots(num_features)
    for i in range(num_features):
        ax = axes[i]
        x = obs[:train_pool._size, i]
        ax.hist(x)
    plt.title("observation")
    plt.show()

    env = SimpleReacherEnv()
    batch = train_pool.random_batch(1)
    computed_rewards = env.compute_rewards(
        batch['observations'],
        batch['actions'],
        batch['next_observations'],
        0.2 * np.ones((1, 2)),
    )
    import ipdb; ipdb.set_trace()
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_pkl_path', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    dataset_path = args.replay_pkl_path
    main(dataset_path)
