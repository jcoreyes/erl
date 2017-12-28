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
from railrl.state_distance.tdm_networks import StructuredQF
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.torch.modules import HuberLoss
from railrl.torch.networks import TanhMlpPolicy


def experiment(variant):
    env = normalize_box(variant['env_class'](**variant['env_kwargs']))

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['ddpg_tdm_kwargs']['tdm_kwargs']['vectorized']
    qf = StructuredQF(
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=env.goal_dim,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + env.goal_dim + 1,
        output_size=action_dim,
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
    exp_prefix = "ant-far-distance-longer"

    num_epochs = 1000
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 50

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
            internal_gcm=True,
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
    )
    search_space = {
        'env_class': [
            # Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            # GoalXPosHalfCheetah,
            GoalXYPosAnt,
            # CylinderXYPusher2DEnv,
            # Walker2DTargetXPos,
            # MultitaskPusher3DEnv,
        ],
        'env_kwargs': [
            # dict(),
            dict(max_distance=2),
            dict(max_distance=4),
            dict(max_distance=6),
            dict(max_distance=8),
            dict(max_distance=10),
        ],
        'qf_criterion_class': [
            nn.MSELoss,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            'environment',
        ],
        'es_kwargs': [
            dict(theta=0.1, max_sigma=0.1, min_sigma=0.1),
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.max_tau': [
            49,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.dense_rewards': [
            False,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.finite_horizon': [
            True,
        ],
        # 'ddpg_tdm_kwargs.tdm_kwargs.sample_train_goals_from': [
        # 'no_resampling',
        # ],
        'ddpg_tdm_kwargs.tdm_kwargs.tau_sample_strategy': [
            'truncated_geometric',
            # 'uniform',
            # 'no_resampling',
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.truncated_geom_factor': [
            1,
        ],
        'ddpg_tdm_kwargs.base_kwargs.reward_scale': [
            1, 100, 10000, int(1e6),
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
        'instance_type': [
            'c5.large',
            # 'g2.2xlarge',
        ],
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
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            instance_type = variant['instance_type']
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                region='us-east-1',
                instance_type=instance_type,
                use_gpu=variant['instance_type'][0] == 'g',
            )
