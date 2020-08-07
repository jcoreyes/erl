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
from rlkit.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from rlkit.envs.multitask.pusher3d import MultitaskPusher3DEnv
from rlkit.envs.multitask.pusher3d_gym import GoalXYGymPusherEnv
from rlkit.envs.multitask.walker2d_env import Walker2DTargetXPos
from rlkit.envs.multitask.reacher_7dof import (
    # Reacher7DofGoalStateEverything,
    Reacher7DofXyzGoalState,
)
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.her import HER, HerQFunction, HerPolicy
from rlkit.state_distance.tdm_networks import TdmNormalizer
from rlkit.torch.modules import HuberLoss
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class'](**variant['env_kwargs']))
    observation_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    obs_normalizer = TorchFixedNormalizer(observation_dim)
    goal_normalizer = TorchFixedNormalizer(env.goal_dim)
    action_normalizer = TorchFixedNormalizer(action_dim)
    distance_normalizer = TorchFixedNormalizer(env.goal_dim)
    tdm_normalizer = TdmNormalizer(
        env,
        obs_normalizer=obs_normalizer,
        goal_normalizer=goal_normalizer,
        action_normalizer=action_normalizer,
        distance_normalizer=distance_normalizer,
        max_tau=1,
        **variant['tdm_normalizer_kwargs']
    )
    qf = HerQFunction(
        env=env,
        **variant['qf_kwargs']
    )
    policy = HerPolicy(
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
    algorithm = HER(
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
    exp_prefix = "dev-her"

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "final-ant-pos-and-vel"

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
                batch_size=256,
                discount=1,
            ),
            tdm_kwargs=dict(),
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
        version="HER-Andrychowicz",
        algorithm="HER-Andrychowicz",
        env_kwargs=dict(),
        tdm_normalizer_kwargs=dict(
            normalize_tau=False,
            log_tau=False,
        ),
    )
    search_space = {
        'env_class': [
            # GoalXVelHalfCheetah,
            # GoalXPosHalfCheetah,
            # GoalXYPosAnt,
            # GoalXYGymPusherEnv,
            # CylinderXYPusher2DEnv,
            # Reacher7DofXyzGoalState,
            # MultitaskPusher3DEnv,
            # Walker2DTargetXPos,
            GoalXYPosAndVelAnt,
        ],
        'env_kwargs.speed_weight': [
            None,
        ],
        'env_kwargs.goal_dim_weights': [
            (0.1, 0.1, 0.9, 0.9),
        ],
        'qf_criterion_class': [
            nn.MSELoss,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            'environment',
        ],
        'her_replay_buffer_kwargs.num_goals_to_sample': [
            4,
        ],
        'ddpg_tdm_kwargs.base_kwargs.reward_scale': [
            1, 10, 100,
        ],
        'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': [
            5, 10
        ],
        'ddpg_tdm_kwargs.base_kwargs.discount': [
            0.98,
        ],
        'ddpg_tdm_kwargs.ddpg_kwargs.tau': [
            0.05,
        ],
        'ddpg_tdm_kwargs.ddpg_kwargs.policy_pre_activation_weight': [
            0.,
            0.01,
        ],
        'ddpg_tdm_kwargs.ddpg_kwargs.eval_with_target_policy': [
            True,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.num_pretrain_paths': [
            20, 0
        ],
        'relabel': [
            False,
            True,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        relabel = variant['relabel']
        if not relabel:
            variant['ddpg_tdm_kwargs']['tdm_kwargs']['sample_train_goals_from'] = 'no_resampling'
            variant['ddpg_tdm_kwargs']['tdm_kwargs']['tau_sample_strategy'] = 'no_resampling'
            variant['version'] = "DDPG-Sparse"
            variant['algorithm'] = "DDPG-Sparse"
        for i in range(n_seeds):
            variant['multitask'] = (
                    variant['ddpg_tdm_kwargs']['tdm_kwargs'][
                        'sample_rollout_goals_from'
                    ] != 'fixed'
            )
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
            )
