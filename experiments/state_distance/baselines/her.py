import argparse
import random

import numpy as np
from hyperopt import hp
from torch import nn as nn
from torch.nn import functional as F

from railrl.algos.state_distance.her import HER
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.pusher2d import MultitaskPusher2DEnv
from railrl.networks.her import HerPolicy, HerQFunction
from railrl.pythonplusplus import identity

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
    HorizonFedStateDistanceQLearning)
from railrl.algos.state_distance.util import get_replay_buffer
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
    UniversalQfunction,
    FlatUniversalQfunction,
    StructuredUniversalQfunction,
    GoalStructuredUniversalQfunction,
    DuelingStructuredUniversalQfunction,
)
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
        env.goal_dim,
        **variant['qf_params']
    )
    policy = FFUniversalPolicy(
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
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
    es = variant['es_class'](
        action_space=action_space,
        **variant['es_params']
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
    replay_buffer = HerReplayBuffer(
        observation_dim=convert_gym_space(env.observation_space).flat_dim,
        action_dim=convert_gym_space(env.action_space).flat_dim,
        **variant['replay_buffer_params'],
    )
    algo = HER(
        env,
        qf,
        policy,
        exploration_policy,
        qf_criterion=qf_criterion,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-baseline-her"
    run_mode = "none"

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "sdql-reacher2d-eval-fix"
    # run_mode = 'grid'

    version = "na"
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
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=10,
            use_soft_update=True,
            tau=0.001,
            batch_size=500,
            discount=5,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            replay_buffer_size=200000,
            render=args.render,
            goal_sample_strategy='store',
        ),
        qf_class=HerQFunction,
        qf_params=dict(
            hidden_sizes=[64, 64, 64],
            hidden_activation=F.softplus,
        ),
        policy_class=HerPolicy,
        policy_params=dict(
            hidden_sizes=[64, 64, 64],
            hidden_activation=F.softplus,
        ),
        replay_buffer_params=dict(
            num_goals_to_sample=4,
            goal_sample_strategy='store',
        ),
        # env_class=Reacher7DofFullGoalState,
        # env_class=ArmEEInStatePusherEnv,
        # env_class=JointOnlyPusherEnv,
        env_class=XyMultitaskSimpleStateReacherEnv,
        # env_class=MultitaskPusher2DEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        # env_class=MultitaskPoint2DEnv,
        env_params=dict(),
        normalize_params=dict(),
        es_class=OUStrategy,
        es_params=dict(
            theta=0.1,
            max_sigma=0.02,
            min_sigma=0.02,
        ),
        generate_data=args.replay_path is None,
        qf_criterion_class=HuberLoss,
        # qf_criterion_class=nn.MSELoss,
        qf_criterion_params=dict(
            # delta=1,
        ),
        exp_prefix=exp_prefix,
    )
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
