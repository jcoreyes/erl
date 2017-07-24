import argparse
import numpy as np
from torch.autograd import Variable
import railrl.torch.pytorch_util as ptu

from pathlib import Path
import random
import pickle
import os
import os.path as osp

from railrl.algos.qlearning.state_distance_q_learning import (
    StateDistanceQLearning,
    StateDistanceQLearningSimple,
)
from railrl.samplers.path_sampler import MultitaskPathSampler
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.reacher_env import (
    MultitaskReacherEnv,
    SimpleReacherEnv,
    GoalStateReacherEnv,
)
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.policies.zero_policy import ZeroPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from rllab.config import LOG_DIR
from rllab.misc import logger
import matplotlib.pyplot as plt


def main(variant):
    env_class = variant['env_class']
    env = env_class()
    dataset_path = variant['dataset_path']
    with open(dataset_path, 'rb') as handle:
        pool = pickle.load(handle)

    # train_pool = pool.train_replay_buffer
    # actions = train_pool._actions
    # obs = train_pool._observations
    # num_features = obs.shape[-1]
    # fig, axes = plt.subplots(num_features)
    # for i in range(num_features):
    #     ax = axes[i]
    #     x = obs[:train_pool._size, i]
    #     ax.hist(x)
    # plt.show()
    #
    # import ipdb; ipdb.set_trace()
    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = FeedForwardQFunction(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
        batchnorm_obs=True,
    )
    policy = FeedForwardPolicy(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
    )
    algo = StateDistanceQLearning(
    # algo = StateDistanceQLearningSimple(
        env=env,
        qf=qf,
        policy=policy,
        # exploration_strategy=es,
        pool=pool,
        exploration_policy=None,
        **variant['algo_params']
    )
    algo.train()
    # qf = algo.qf
    # goal = np.array([.2, .2])
    # num_samples = 100
    #
    # obs = env.reset()
    # for _ in range(1000):
    #     new_obs = np.hstack((obs, goal))
    #     action = sample_best_action(qf, new_obs, num_samples)
    #     obs, r, d, env_info = env.step(action)
    #     env.render()
    # import ipdb; ipdb.set_trace()
    # print("done")


def sample_best_action(qf, obs, num_samples):
    sampled_actions = np.random.uniform(-.1, .1, size=(num_samples, 2))
    obs_expanded = np.repeat(np.expand_dims(obs, 0), num_samples, axis=0)
    actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
    obs = Variable(ptu.from_numpy(obs_expanded).float(), requires_grad=False)
    q_values = ptu.get_numpy(qf(obs, actions))
    max_i = np.argmax(q_values)
    return sampled_actions[max_i]


def grid_search_best_action(qf, obs, resolution):
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    sampled_actions = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    num_samples = resolution**2
    obs_expanded = np.repeat(np.expand_dims(obs, 0), num_samples, axis=0)
    actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
    obs = Variable(ptu.from_numpy(obs_expanded).float(), requires_grad=False)
    q_values = ptu.get_numpy(qf(obs, actions))
    max_i = np.argmax(q_values)
    # vals = q_values.reshape(resolution, resolution)
    # heatmap = vals, x, y, _
    # fig, ax = plt.subplots(1, 1)
    # plot_heatmap(fig, ax, heatmap)
    # plt.show()
    return sampled_actions[max_i]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_pkl_path', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    n_seeds = 1
    mode = "here"
    # exp_prefix = "7-17-dev-state-distance-train-multiple-goals-nonzero-gamma"
    exp_prefix = "7-18-dev-sdql-train-goal-state-reacher-env-10k-only-joints"
    snapshot_mode = 'all'

    # out_dir = Path(LOG_DIR) / 'datasets/generated'
    # out_dir /= '7-10-reacher'
    # logger.set_snapshot_dir(str(out_dir))
    dataset_path = args.replay_pkl_path

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_batches=100000,
            num_batches_per_epoch=1000,
            use_soft_update=True,
            tau=1e-3,
            batch_size=1024,
            discount=0.,
            qf_learning_rate=1e-4,
            policy_learning_rate=1e-5,
        ),
        env_class=GoalStateReacherEnv,
        # env_class=SimpleReacherEnv,
    )

    seed = random.randint(0, 10000)
    run_experiment(
        main,
        exp_prefix=exp_prefix,
        seed=seed,
        mode=mode,
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode=snapshot_mode,
    )
