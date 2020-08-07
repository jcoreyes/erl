import random

import numpy as np
import torch.nn as nn

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.ant_env import GoalXYPosAnt
# from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from rlkit.envs.multitask.pusher3d import MultitaskPusher3DEnv
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
from rlkit.state_distance.experimental_tdm_networks import StructuredQF
from rlkit.state_distance.tdm_ddpg import TdmDdpg
from rlkit.torch.modules import HuberLoss
from rlkit.torch.networks import TanhMlpPolicy, FeedForwardPolicy


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())

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
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-ddpg-dense-her-launch"

    n_seeds = 2
    mode = "ec2"
    exp_prefix = "tdm-ant"

    num_epochs = 200
    num_steps_per_epoch = 10000
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
                discount=0.98,
            ),
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=True,
                cycle_taus_for_rollout=True,
                max_tau=10,
                dense_rewards=True,
            ),
            ddpg_kwargs=dict(
                tau=0.001,
                qf_learning_rate=1e-3,
                policy_learning_rate=1e-4,
            ),
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(2E5),
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
        version="HER-Dense DDPG",
        algorithm="HER-Dense DDPG",
    )
    search_space = {
        'env_class': [
            # Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            # GoalXPosHalfCheetah,
            GoalXYPosAnt,
            # Walker2DTargetXPos,
            # MultitaskPusher3DEnv,
        ],
        'qf_criterion_class': [
            nn.MSELoss,
            # HuberLoss,
        ],
        # 'es_kwargs': [
            # dict(theta=0.1, max_sigma=0.1, min_sigma=0.1),
            # dict(theta=0.01, max_sigma=0.1, min_sigma=0.1),
            # dict(theta=0.1, max_sigma=0.05, min_sigma=0.05),
            # dict(theta=0.1, max_sigma=0.2, min_sigma=0.2),
        # ],
        'ddpg_tdm_kwargs.tdm_kwargs.sample_rollout_goals_from': [
            # 'fixed',
            'environment',
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.max_tau': [
            0
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.dense_rewards': [
            True,
        ],
        'ddpg_tdm_kwargs.tdm_kwargs.finite_horizon': [
            False,
        ],
        'ddpg_tdm_kwargs.base_kwargs.reward_scale': [
            0.01, 1, 100,
        ],
        'ddpg_tdm_kwargs.base_kwargs.discount': [
            0.98, 0.95
        ],
        'ddpg_tdm_kwargs.base_kwargs.num_updates_per_env_step': [
            1,
        ],
        # 'ddpg_tdm_kwargs.ddpg_kwargs.tau': [
            # 0.001,
            # 0.01,
        # ],
        'ddpg_tdm_kwargs.ddpg_kwargs.eval_with_target_policy': [
            # True,
            False,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
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
