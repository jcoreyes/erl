"""
Example of running NAF on HalfCheetah.
"""

import railrl.torch.pytorch_util as ptu
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import setup_logger
from railrl.torch.algos.naf import NafPolicy, NAF
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize


def experiment(variant):
    env = normalize(HalfCheetahEnv())
    es = OUStrategy(action_space=env.action_space)
    policy = NafPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = NAF(
        env,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=10000,
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
