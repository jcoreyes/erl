import argparse
import random

import numpy as np
from hyperopt import hp
from torch import nn as nn
from torch.nn import functional as F

from rlkit.envs.multitask.pusher2d import HandCylinderXYPusher2DEnv

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.algos.state_distance.state_distance_q_learning import (
    HorizonFedStateDistanceQLearning)
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.multitask.reacher_7dof import (
    Reacher7DofFullGoalState,
)
from rlkit.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
)
from rlkit.envs.multitask.pusher import (
    JointOnlyPusherEnv,
)
from rlkit.envs.wrappers import convert_gym_space, NormalizedBoxEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import (
    create_log_dir,
    create_run_experiment_multiple_seeds,
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc.hypopt import optimize_and_save
from rlkit.misc.ml_util import ConstantSchedule
from rlkit.networks.state_distance import (
    FFUniversalPolicy,
    GoalStructuredUniversalQfunction,
)
from rlkit.state_distance.policies import TerminalRewardSampleOCPolicy
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    env = NormalizedBoxEnv(
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
    exp_prefix = "hyper-param-sweep"
    run_mode = "none"

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "sdql-reacher2d-eval-fix"
    # run_mode = 'grid'

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
            num_steps_per_epoch=100,
            num_steps_per_eval=100,
            num_updates_per_env_step=10,
            use_soft_update=True,
            tau=0.001,
            batch_size=64,
            discount=5,
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
            cycle_taus_for_rollout=True,
        ),
        explore_with_ddpg_policy=True,
        # qf_class=UniversalQfunction,
        # qf_class=FlatUniversalQfunction,
        # qf_class=StructuredUniversalQfunction,
        qf_class=GoalStructuredUniversalQfunction,
        # qf_class=DuelingStructuredUniversalQfunction,
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.softplus,
            # output_activation=F.softplus,
            # output_multiplier=-1,
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        # epoch_discount_schedule_class=IntRampUpSchedule,
        epoch_discount_schedule_class=ConstantSchedule,
        epoch_discount_schedule_params=dict(
            value=5,
            # min_value=0,
            # max_value=100,
            # ramp_duration=50,
        ),
        algo_class=HorizonFedStateDistanceQLearning,
        env_class=Reacher7DofFullGoalState,
        # env_class=ArmEEInStatePusherEnv,
        # env_class=JointOnlyPusherEnv,
        # env_class=GoalStateSimpleStateReacherEnv,
        # env_class=MultitaskPusher2DEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        # env_class=MultitaskPoint2DEnv,
        env_params=dict(),
        normalize_params=dict(
            # obs_mean=None,
            # obs_std=[0.7, 0.7, 0.7, 0.6, 40, 5],
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
    if run_mode == 'grid':
        search_space = {
            'env_class': [
                JointOnlyPusherEnv,
                Reacher7DofFullGoalState,
                GoalStateSimpleStateReacherEnv,
                HandCylinderXYPusher2DEnv,
                MultitaskPoint2DEnv,
            ],
            # 'env_class': [
            #     Reacher7DofFullGoalState,
            #     GoalStateSimpleStateReacherEnv,
            #     # ArmEEInStatePusherEnv,
            #     # JointOnlyPusherEnv,
            # ],
            # 'qf_class': [StructuredUniversalQfunction, FlatUniversalQfunction],
            # 'epoch_discount_schedule_params.value': [5, 10, 50],
            # 'algo_params.sparse_reward': [True, False],
            # 'algo_params.clamp_q_target_values': [True, False],
            # 'algo_params.prob_goal_state_is_next_state': [0.5, 0],
            # 'qf_params.dropout_prob': [0.5, 0],
            # 'algo_params.qf_weight_decay': [1e-3, 1e-4, 1e-5, 0],
            # 'algo_params.sample_goals_from': ['environment', 'replay_buffer'],
            # 'algo_params.sample_discount': [True, False],
            # 'algo_params.num_steps_per_epoch': [1, 10],
            # 'algo_params.termination_threshold': [1e-4, 0]
            # 'algo_params.fraction_of_taus_set_to_zero': [0.5, 0],
            # 'algo_params.optimize_target_policy': [True, False],
            # 'algo_params.residual_gradient_weight': [0.5, 0],
            # 'epoch_discount_schedule_params.max_value': [100, 1000],
            # 'epoch_discount_schedule_params.ramp_duration': [
            #     1, 20, 50, 200,
            # ],
            # 'qf_params.output_activation': [
            #     identity,
            #     F.softplus,
            # ]
            # 'qf_params': [
            #     dict(
            #         obs_hidden_size=400,
            #         embed_hidden_size=300,
            #     ),
            #     dict(
            #         obs_hidden_size=100,
            #         embed_hidden_size=100,
            #     ),
            # ],
            # 'policy_params': [
            #     dict(
            #         fc1_size=400,
            #         fc2_size=300,
            #     ),
            #     dict(
            #         fc1_size=100,
            #         fc2_size=100,
            #     ),
            # ],
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
    elif run_mode == 'custom_grid':
        for exp_id, (
                nupo,
                num_steps_per_epoch,
                version,
        ) in enumerate([
            (1, 10000, "500k_env_steps"),
            (10, 500, "50k_env_steps"),
            (50, 100, "10k_env_steps"),
        ]):
            variant['algo_params']['num_updates_per_env_step'] = nupo
            variant['algo_params']['num_steps_per_epoch'] = num_steps_per_epoch
            variant['version'] = version
            # search_space = {
            #     'algo_params.sample_goals_from': ['environment', 'replay_buffer'],
            #     'explore_with_ddpg_policy': [True, False],
            #     'normalize_params.obs_std': [
            #         [0.7, 0.7, 0.7, 0.6, 40, 5],
            #         None,
            #     ],
            # }
            # sweeper = hyp.DeterministicHyperparameterSweeper(
            #     search_space, default_parameters=variant.copy(),
            # )
            # for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                seed = random.randint(0, 10000)
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    elif run_mode == 'hyperopt':
        search_space = {
            'float_param': hp.uniform(
                'float_param',
                0.,
                5,
            ),
            'float_param2': hp.loguniform(
                'float_param2',
                np.log(0.01),
                np.log(1000),
            ),
            'seed': hp.randint('seed', 10000),
        }

        base_log_dir = create_log_dir(exp_prefix=exp_prefix)

        optimize_and_save(
            base_log_dir,
            create_run_experiment_multiple_seeds(
                n_seeds,
                experiment,
                exp_prefix=exp_prefix,
            ),
            search_space=search_space,
            extra_function_kwargs=variant,
            maximize=True,
            verbose=True,
            load_trials=True,
            num_rounds=500,
            num_evals_per_round=1,
        )
    elif run_mode == 'random':
        hyperparameters = [
            # hyp.EnumParam('qf_params.dropout', [True, False]),
            hyp.EnumParam('algo_params.qf_criterion_class', [
                HuberLoss,
                nn.MSELoss,
            ]),
            hyp.EnumParam('qf_params.hidden_sizes', [
                [100, 100],
                [800, 600, 400],
            ]),
            hyp.LogFloatParam('algo_params.qf_weight_decay', 1e-5, 1e-2),
        ]
        sweeper = hyp.RandomHyperparameterSweeper(
            hyperparameters,
            default_kwargs=variant,
        )
        for _ in range(num_configurations):
            for exp_id in range(n_seeds):
                seed = random.randint(0, 10000)
                variant = sweeper.generate_random_hyperparameters()
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
