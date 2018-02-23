"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
import gym

from railrl.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.state_distance.tdm_supervised import TdmSupervised
import railrl.torch.pytorch_util as ptu
from railrl.state_distance.tdm_networks import TdmPolicy

def experiment(variant):
    env = NormalizedBoxEnv(Reacher7DofXyzGoalState())
    es = OUStrategy(action_space=env.action_space)
    policy = TdmPolicy(
        env=env,
        **variant['policy_kwargs']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TdmSupervised(
        env,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['supervised_tdm_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        supervised_tdm_kwargs=dict(
            base_kwargs=dict(
                num_epochs=100,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=99,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
            ),
            tdm_kwargs=dict(
                max_tau=10,
            ),
        )
    )
    setup_logger('TEST', variant=variant)
    experiment(variant)
