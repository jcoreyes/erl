from pathlib import Path
import pickle
import os
import os.path as osp

from railrl.algos.qlearning.state_distance_q_learning import (
    StateDistanceQLearning,
    MultitaskPathSampler, StateDistanceQLearningSimple)
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.reacher_env import MultitaskReacherEnv
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.policies.torch import FeedForwardPolicy
from railrl.policies.zero_policy import ZeroPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from rllab.config import LOG_DIR


def main(variant):
    env = MultitaskReacherEnv()
    action_space = convert_gym_space(env.action_space)
    dataset_path = variant['dataset_path']
    with open(dataset_path, 'rb') as handle:
        pool = pickle.load(handle)

    observation_space = convert_gym_space(env.observation_space)
    qf = FeedForwardQFunction(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
        )
    policy = FeedForwardPolicy(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
        )
    algo = StateDistanceQLearningSimple(
        env=env,
        qf=qf,
        policy=policy,
        pool=pool,
        exploration_policy=None,
        **variant['algo_params']
    )
    algo.train()

if __name__ == '__main__':
    out_dir = Path(LOG_DIR) / 'datasets/generated'
    out_dir /= '7-10-reacher'
    dataset_path = out_dir / 'data.pkl'

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_batches=100,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            discount=0.,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        pool_size=1000000,
    )
    main(variant)
