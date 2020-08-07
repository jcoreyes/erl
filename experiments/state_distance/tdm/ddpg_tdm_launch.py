import random

import numpy as np
import torch.nn as nn

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.ant_env import GoalXYPosAnt, GoalXYPosAndVelAnt
# from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from rlkit.envs.multitask.point2d import MultitaskPoint2DEnv
from rlkit.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from rlkit.envs.multitask.pusher3d import MultitaskPusher3DEnv
from rlkit.envs.multitask.pusher3d_gym import GoalXYGymPusherEnv
from rlkit.envs.multitask.walker2d_env import Walker2DTargetXPos
from rlkit.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofXyzGoalState,
    Reacher7DofXyzPosAndVelGoalState, Reacher7DofFullGoal)
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.tdm_networks import TdmPolicy, \
    TdmQf, TdmNormalizer
from rlkit.state_distance.experimental_tdm_networks import StructuredQF, \
    InternalGcmQf
from rlkit.state_distance.tdm_ddpg import TdmDdpg
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import HuberLoss
from rlkit.torch.networks import TanhMlpPolicy, FeedForwardPolicy


def experiment(variant):
    vectorized = variant['vectorized']
    norm_order = variant['norm_order']

    variant['ddpg_tdm_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['ddpg_tdm_kwargs']['tdm_kwargs']['norm_order'] = norm_order

    env = NormalizedBoxEnv(variant['env_class'](**variant['env_kwargs']))
    max_tau = variant['ddpg_tdm_kwargs']['tdm_kwargs']['max_tau']
    tdm_normalizer = TdmNormalizer(
        env,
        vectorized,
        max_tau=max_tau,
        **variant['tdm_normalizer_kwargs']
    )
    qf = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        tdm_normalizer=tdm_normalizer,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
        tdm_normalizer=tdm_normalizer,
        **variant['policy_kwargs']
    )
    es = OUStrategy(
        action_space=env.action_space,
        **variant['es_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_kwargs']
    )
    qf_criterion = variant['qf_criterion_class'](
        **variant['qf_criterion_kwargs']
    )
    ddpg_tdm_kwargs = variant['ddpg_tdm_kwargs']
    ddpg_tdm_kwargs['ddpg_kwargs']['qf_criterion'] = qf_criterion
    ddpg_tdm_kwargs['tdm_kwargs']['tdm_normalizer'] = tdm_normalizer
    algorithm = TdmDdpg(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['ddpg_tdm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-ddpg-tdm-launch-2"

    # n_seeds = 1
    # mode = "ec2"
    # exp_prefix = "reacher-full-ddpg-tdm-mtau-0"

    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 100

    # noinspection PyTypeChecker
    variant = dict(
        ddpg_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=25,
                batch_size=128,
                discount=1,
                collection_mode='online',
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
                cycle_taus_for_rollout=True,
                max_tau=10,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        qf_criterion_class=HuberLoss,
        qf_criterion_kwargs=dict(),
        version="DDPG-TDM",
        algorithm="DDPG-TDM",
        tdm_normalizer_kwargs=dict(
            normalize_tau=False,
            log_tau=False,
        ),
        env_kwargs=dict(),
    )
    search_space = {
        'env_class': [
            # GoalXVelHalfCheetah,
            # GoalXPosHalfCheetah,
            # GoalXYPosAnt,
            # Reacher7DofXyzPosAndVelGoalState,
            # GoalXYPosAndVelAnt,
            # MultitaskPusher3DEnv,
            # Reacher7DofXyzGoalState,
            # CylinderXYPusher2DEnv,
            # Walker2DTargetXPos,
            # GoalXYGymPusherEnv,
            # Reacher7DofFullGoal,
            MultitaskPoint2DEnv,
        ],
        # 'env_kwargs.max_distance': [
        #     6,
        # ],
        # 'env_kwargs.min_distance': [
        #     3,
        # ],
        # 'env_kwargs.speed_weight': [
        #     None,
        # ],
        # 'env_kwargs.goal_dim_weights': [
        #     (0.1, 0.1, 0.9, 0.9),
        # ],
        'tdm_normalizer_kwargs.log_tau': [
            False,
        ],
        'tdm_normalizer_kwargs.normalize_tau': [
            False,
        ],
        'qf_criterion_class': [
            # HuberLoss,
            nn.MSELoss,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            # 'environment',
            'pretrain_paths',
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.num_pretrain_paths': [
            20
        ],
        'es_kwargs': [
            dict(theta=0.1, max_sigma=0.1, min_sigma=0.1),
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.max_tau': [
            10,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.dense_rewards': [
            False,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.finite_horizon': [
            True,
        ],
        'relabel': [
            True,
        ],
        'her_replay_buffer_kwargs.resampling_strategy': [
            # 'truncated_geometric',
            'uniform',
        ],
        'her_replay_buffer_kwargs.num_goals_to_sample': [
            4,
        ],
        'her_replay_buffer_kwargs.truncated_geom_factor': [
            1,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.tau_sample_strategy': [
            'uniform',
            # 'truncated_geometric',
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.truncated_geom_factor': [
            1,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.reward_type': [
            'distance',
        ],
        'qf_kwargs.structure': [
            'norm_difference',
            # 'norm',
            # 'norm_distance_difference',
            # 'distance_difference',
            # 'difference',
            # 'none',
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.terminate_when_goal_reached': [
            False
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.normalize_distance': [
            False
        ],
        'ddpg_tdm_kwargs.base_kwargs.reward_scale': [
            # 0.01, 1, 100, 10000
            1,
        ],
        'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': [
            1,
        ],
        'ddpg_tdm_kwargs.base_kwargs.discount': [
            1,
        ],
        'ddpg_tdm_kwargs.base_kwargs.batch_size': [
            128,
        ],
        'ddpg_tdm_kwargs.ddpg_kwargs.eval_with_target_policy': [
            False,
        ],
        'vectorized': [True],
        'norm_order': [1],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant['multitask'] = (
                variant['ddpg_tdm_kwargs']['tdm_kwargs'][
                    'sample_rollout_goals_from'
                ] != 'fixed'
        )
        dense = variant['ddpg_tdm_kwargs']['tdm_kwargs']['dense_rewards']
        finite = variant['ddpg_tdm_kwargs']['tdm_kwargs']['finite_horizon']
        relabel = variant['relabel']
        vectorized = variant['vectorized']
        variant['ddpg_tdm_kwargs']['tdm_kwargs']['vectorized'] = vectorized
        norm_order = variant['norm_order']

        # some settings just don't make sense
        if vectorized and norm_order != 1:
            continue
        if not dense and not finite:
            continue
        if not finite:
            # For infinite case, max_tau doesn't matter, so just only run for
            # one setting of max tau
            if variant['ddpg_tdm_kwargs']['tdm_kwargs']['max_tau'] != (
                max_path_length - 1
            ):
                continue
            discount = variant['ddpg_tdm_kwargs']['base_kwargs']['discount']
            variant['ddpg_tdm_kwargs']['base_kwargs']['discount'] = min(
                0.98, discount
            )
        if not relabel:
            variant['ddpg_tdm_kwargs']['tdm_kwargs']['sample_train_goals_from'] = 'no_resampling'
            variant['ddpg_tdm_kwargs']['tdm_kwargs']['tau_sample_strategy'] = 'no_resampling'
        use_gpu = (
            variant['ddpg_tdm_kwargs']['base_kwargs']['batch_size'] == 1024
        )
        for i in range(n_seeds):
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=exp_id,
                use_gpu=use_gpu,
            )
