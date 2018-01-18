"""
HER implementation.

I know it says "TDM" but I coded up TDM so that HER is a specific configuration
of TDMs.
"""

import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState,
)
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.state_distance.tdm_networks import TdmQf, TdmPolicy, TdmNormalizer
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.modules import HuberLoss


def experiment(variant):
    vectorized = variant['ddpg_tdm_kwargs']['tdm_kwargs']['vectorized']
    assert not vectorized
    env = NormalizedBoxEnv(Reacher7DofXyzGoalState())
    observation_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    obs_normalizer = TorchFixedNormalizer(observation_dim)
    goal_normalizer = TorchFixedNormalizer(env.goal_dim)
    action_normalizer = TorchFixedNormalizer(action_dim)
    distance_normalizer = TorchFixedNormalizer(
        env.goal_dim if vectorized else 1
    )
    max_tau = variant['ddpg_tdm_kwargs']['tdm_kwargs']['max_tau']
    tdm_normalizer = TdmNormalizer(
        env,
        obs_normalizer=obs_normalizer,
        goal_normalizer=goal_normalizer,
        action_normalizer=action_normalizer,
        distance_normalizer=distance_normalizer,
        max_tau=max_tau,
    )
    qf = TdmQf(
        env=env,
        vectorized=vectorized,
        norm_order=2,
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
    qf_criterion = HuberLoss()
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
                num_updates_per_env_step=1,
                batch_size=256,
                discount=0.95,
            ),
            # These three configuration make it HER and not TDM
            tdm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                vectorized=False,
                cycle_taus_for_rollout=False,
                max_tau=0,
                finite_horizon=False,
                dense_rewards=True,
                reward_type='indicator',
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
            # This is also needed for make the QF a normal QF (and not a TDM)
            structure='none',
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        es_kwargs=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        algorithm="HER",
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
