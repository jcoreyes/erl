import argparse
import random

import numpy as np
from hyperopt import hp
from torch import nn as nn
from torch.nn import functional as F


import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.vectorized_sdql import VectorizedSdql, \
    VectorizedTauSdql

from railrl.envs.multitask.pusher2d import MultitaskPusher2DEnv
from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState,
    Reacher7DofFullGoalState,
    Reacher7DofCosSinFullGoalState,
)
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
    XyMultitaskSimpleStateReacherEnv,
)
from railrl.envs.multitask.pusher import (
    ArmEEInStatePusherEnv,
    JointOnlyPusherEnv,
)

from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import (
    create_log_dir,
    create_run_experiment_multiple_seeds,
)
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.hypopt import optimize_and_save
from railrl.misc.ml_util import RampUpSchedule, IntRampUpSchedule, \
    ConstantSchedule
from railrl.networks.state_distance import (
    FFUniversalPolicy,
    VectorizedGoalStructuredUniversalQfunction)
from railrl.policies.state_distance import TerminalRewardSampleOCPolicy
from railrl.torch.modules import HuberLoss
from railrl.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    env = normalize_box(
        env,
        **variant['normalize_params']
    )

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = variant['qf_class'](
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        int(observation_space.flat_dim),
        **variant['qf_params']
    )
    policy = FFUniversalPolicy(
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        int(observation_space.flat_dim),
        **variant['policy_params']
    )
    epoch_discount_schedule = None
    epoch_discount_schedule_class = variant['epoch_discount_schedule_class']
    if epoch_discount_schedule_class is not None:
        epoch_discount_schedule = epoch_discount_schedule_class(
            **variant['epoch_discount_schedule_params']
        )
    qf_criterion = variant['qf_criterion_class'](
        **variant['qf_criterion_params']
    )
    es = variant['sampler_es_class'](
        action_space=action_space,
        **variant['sampler_es_params']
    )
    if variant['explore_with_ddpg_policy']:
        raw_exploration_policy = policy
    else:
        raw_exploration_policy = TerminalRewardSampleOCPolicy(
            qf,
            env,
            5,
        )
    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=raw_exploration_policy,
    )
    algo = variant['algo_class'](
        env,
        qf,
        policy,
        exploration_policy,
        epoch_discount_schedule=epoch_discount_schedule,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-vectorized-train-q"
    run_mode = "none"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "vectorized-train-many-envs-lots-of-data"
    # exp_prefix = "vectorized-train-q-env"
    run_mode = 'grid'

    version = "na"
    num_configurations = 50  # for random mode
    snapshot_mode = "last"
    snapshot_gap = 10
    use_gpu = True
    if mode != "local":
        use_gpu = False

    dataset_path = args.replay_path

    max_path_length = 100
    max_tau = 10
    # noinspection PyTypeChecker
    variant = dict(
        version=version,
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_epochs=101,
            num_steps_per_epoch=10000,
            num_steps_per_eval=10000,
            num_updates_per_env_step=10,
            use_soft_update=True,
            tau=0.001,
            batch_size=500,
            discount=max_tau,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            # sample_goals_from='environment',
            sample_goals_from='replay_buffer',
            sample_discount=True,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            replay_buffer_size=200000,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            render=args.render,
            save_replay_buffer=True,
            sparse_reward=True,
            cycle_taus_for_rollout=True,
        ),
        explore_with_ddpg_policy=True,
        qf_class=VectorizedGoalStructuredUniversalQfunction,
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.softplus,
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        # epoch_discount_schedule_class=IntRampUpSchedule,
        epoch_discount_schedule_class=ConstantSchedule,
        epoch_discount_schedule_params=dict(
            # value=0.99,
            value=max_tau,
            # min_value=0,
            # max_value=100,
            # ramp_duration=50,
        ),
        # algo_class=VectorizedSdql,
        algo_class=VectorizedTauSdql,
        # env_class=Reacher7DofFullGoalState,
        # env_class=ArmEEInStatePusherEnv,
        # env_class=JointOnlyPusherEnv,
        # env_class=GoalStateSimpleStateReacherEnv,
        env_class=MultitaskPusher2DEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        # env_class=MultitaskPoint2DEnv,
        env_params=dict(),
        normalize_params=dict(
            # obs_mean=None,
            # obs_std=[1, 1, 1, 1, 20, 20],
        ),
        sampler_params=dict(
            min_num_steps_to_collect=100000,
            max_path_length=max_path_length,
            render=False,
        ),
        sampler_es_class=OUStrategy,
        # sampler_es_class=GaussianStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_params=dict(),
        exp_prefix=exp_prefix,
    )
    if run_mode == 'grid':
        search_space = {
            'env_class': [
                JointOnlyPusherEnv,
                Reacher7DofFullGoalState,
                GoalStateSimpleStateReacherEnv,
                MultitaskPusher2DEnv,
                MultitaskPoint2DEnv,
            ],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant,
        )
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
            )