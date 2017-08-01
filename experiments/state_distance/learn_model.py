import argparse
import pickle
import random

import numpy as np
from hyperopt import hp

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.model_learning import ModelLearning
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
)
from railrl.algos.state_distance.supervised_learning import SupervisedLearning
from railrl.algos.state_distance.util import get_replay_buffer
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
    XyMultitaskSimpleStateReacherEnv)
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import (
    create_log_dir,
    create_run_experiment_multiple_seeds,
)
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.hypopt import optimize_and_save
from railrl.misc.ml_util import RampUpSchedule
from railrl.networks.state_distance import UniversalQfunction
from railrl.policies.zero_policy import ZeroPolicy
from railrl.predictors.torch import Mlp
from railrl.samplers.path_sampler import MultitaskPathSampler


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    replay_buffer = get_replay_buffer(variant)

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    model = Mlp(
        int(observation_space.flat_dim) + int(action_space.flat_dim),
        int(observation_space.flat_dim),
        **variant['model_params']
    )
    algo = ModelLearning(
        env,
        model,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    n_seeds = 1
    mode = "here"
    exp_prefix = "reacher-model-learning"
    version = "Dev"
    run_mode = "none"

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "tmp"

    # run_mode = 'grid'
    num_configurations = 1  # for random mode
    snapshot_mode = "last"
    snapshot_gap = 5
    use_gpu = True
    if mode != "here":
        use_gpu = False

    dataset_path = args.replay_path

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_epochs=101,
            num_batches_per_epoch=1000,
            num_unique_batches=1000,
            batch_size=100,
            learning_rate=1e-4,
        ),
        model_params=dict(
            hidden_sizes=[400, 300],
        ),
        env_class=GoalStateSimpleStateReacherEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        env_params=dict(
            add_noop_action=False,
        ),
        sampler_params=dict(
            min_num_steps_to_collect=20000,
            max_path_length=1000,
            render=False,
        ),
        generate_data=args.replay_path is None,
    )
    seed = random.randint(0, 10000)
    run_experiment(
        experiment,
        exp_prefix=exp_prefix,
        seed=seed,
        mode=mode,
        variant=variant,
        exp_id=0,
        use_gpu=use_gpu,
        sync_s3_log=True,
        sync_s3_pkl=True,
        periodic_sync_interval=3600,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
    )
