"""
Example of running NAF on HalfCheetah.
"""
import gym

import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.torch.naf.naf import NafPolicy, NAF


def experiment(variant):
    env = NormalizedBoxEnv(gym.make('InvertedPendulum-v1'))
    es = OUStrategy(action_space=env.action_space)
    policy = NafPolicy(
        env.observation_space.low.size,
        env.action_space.low.size,
        300,
    )
    target_policy = NafPolicy(
        env.observation_space.low.size,
        env.action_space.low.size,
        300,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = NAF(
        env,
        policy=policy,
        target_policy=target_policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            policy_learning_rate=1e-4,
        ),
        algorithm="NAF",
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
