import random

import numpy as np
import torch.nn as nn

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.ant_env import GoalXYPosAnt
# from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.pusher3d import MultitaskPusher3DEnv
from railrl.envs.multitask.walker2d_env import Walker2DTargetXPos
from railrl.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofXyzGoalState,
)
from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.state_distance.tdm_networks import StructuredQF, TdmPolicy, \
    InternalGcmQf, TdmQf
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.torch.modules import HuberLoss
from railrl.torch.networks import TanhMlpPolicy


def experiment(variant):
    vectorized = variant['vectorized']
    norm_order = variant['norm_order']

    variant['ddpg_tdm_kwargs']['tdm_kwargs']['vectorized'] = vectorized
    variant['ddpg_tdm_kwargs']['tdm_kwargs']['norm_order'] = norm_order
    # variant['env_kwargs']['norm_order'] = norm_order
    env = normalize_box(variant['env_class'](**variant['env_kwargs']))
    qf = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=norm_order,
        **variant['qf_kwargs']
    )
    policy = TdmPolicy(
        env=env,
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
    algorithm = TdmDdpg(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['ddpg_tdm_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-ddpg-tdm-launch"

    n_seeds = 1
    mode = "ec2"
    exp_prefix = "cheetah-xpos-increase-distance"

    num_epochs = 500
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
        version="DDPG-TDM-no-crash",
        algorithm="DDPG-TDM",
    )
    search_space = {
        'env_class': [
            # Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            GoalXPosHalfCheetah,
            # GoalXYPosAnt,
            # CylinderXYPusher2DEnv,
            # Walker2DTargetXPos,
            # MultitaskPusher3DEnv,
        ],
        'env_kwargs': [
            # dict(
                # reward_coefs=(1, 0, 0),
            # ),
            # dict(
            #     reward_coefs=(0.5, 0.375, 0.125),
            # ),
            # dict(max_distance=2),
            # dict(max_distance=4),
            # dict(max_distance=6),
            # dict(max_distance=8),
            dict(max_distance=20),
            dict(max_distance=30),
            dict(max_distance=40),
        ],
        'qf_criterion_class': [
            # HuberLoss,
            nn.MSELoss,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            'environment',
        ],
        'es_kwargs': [
            dict(theta=0.1, max_sigma=0.1, min_sigma=0.1),
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.max_tau': [
            max_path_length-1, 49, 15
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.dense_rewards': [
            False, True,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.finite_horizon': [
            True, False,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.tau_sample_strategy': [
            'uniform',
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.reward_type': [
            'distance',
        ],
        'relabel': [
            True, False,
        ],
        # 'ddpg_tdm_kwargs.tdm_kwargs.truncated_geom_factor': [
        #     1,
        # ],
        'qf_kwargs.structure': [
            'norm_difference',
            # 'norm',
            # 'norm_distance_difference',
            # 'distance_difference',
            # 'difference',
            # 'none',
        ],
        'ddpg_tdm_kwargs.base_kwargs.reward_scale': [
            0.01, 1, 100, 10000,
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
        'ddpg_tdm_kwargs.ddpg_kwargs.tau': [
            0.001,
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
        if relabel:
            variant['ddpg_tdm_kwargs']['tdm_kwargs']['sample_train_goals_from'] = 'her'
            variant['ddpg_tdm_kwargs']['tdm_kwargs']['tau_sample_strategy'] = 'uniform'
        else:
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
