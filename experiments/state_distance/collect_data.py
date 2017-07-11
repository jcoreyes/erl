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
    es = OUStrategy(action_space=action_space)
    exploration_policy = ZeroPolicy(
        int(action_space.flat_dim),
    )
    dataset_path = variant['dataset_path']
    pool_size = variant['pool_size']
    pool = SplitReplayBuffer(
        EnvReplayBuffer(
            pool_size,
            env,
            flatten=True,
        ),
        EnvReplayBuffer(
            pool_size,
            env,
            flatten=True,
        ),
        fraction_paths_in_train=0.8,
    )
    sampler = MultitaskPathSampler(
        env,
        exploration_strategy=es,
        exploration_policy=exploration_policy,
        pool=pool,
        **variant['algo_params']
    )
    sampler.collect_data()
    sampler.save_pool(str(out_dir / 'data.pkl'))
    pool = sampler.pool


if __name__ == '__main__':
    out_dir = Path(LOG_DIR) / 'datasets/generated'
    out_dir /= '7-10-reacher'
    min_num_steps_to_collect = 100000
    max_path_length = 1000
    pool_size = min_num_steps_to_collect + max_path_length

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(out_dir),
        algo_params=dict(
            min_num_steps_to_collect=min_num_steps_to_collect,
            max_path_length=max_path_length,
        ),
        pool_size=pool_size,
    )
    main(variant)
