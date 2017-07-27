"""
Run DDPG on many environments
"""
import random

from railrl.algos.ddpg import DDPG
from railrl.launchers.launcher_util import (
    run_experiment,
    get_standard_env_ids,
    get_env_settings,
)
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.envs.normalized_env import normalize


def example(variant):
    # env_settings = get_env_settings(variant['env_id'])
    # env = env_settings['env']
    env_class = variant['env_class']
    env = env_class()
    env = normalize(env)
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        # n_epochs=5,
        # epoch_length=1000,
        # batch_size=32,
        n_epochs=100,
        epoch_length=10000,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    # for env_id in get_standard_env_ids():
    for env_class in [
        # SwimmerEnv,
        HalfCheetahEnv,
        # AntEnv,
        # HopperEnv,
    ]:
        # for _ in range(5):
        for _ in range(1):
            seed = random.randint(0, 100000)
            run_experiment(
                example,
                exp_prefix="7-26-ddpg-benchmark-timeit-half-cheetah-c4-2xlarge",
                seed=seed,
                mode='ec2',
                # mode='here',
                variant={
                    'env_class': env_class,
                    'version': str(env_class),
                }
            )
