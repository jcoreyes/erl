"""
Run DDPG on many environments
"""
import random

from railrl.tf.ddpg import DDPG
from railrl.launchers.launcher_util import (
    run_experiment,
)
from railrl.policies.tensorflow.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.exploration_strategies.ou_strategy import OUStrategy


def example(variant):
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
        n_epochs=100,
        epoch_length=10000,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    for env_class in [
        SwimmerEnv,
        HalfCheetahEnv,
        AntEnv,
        HopperEnv,
    ]:
        for _ in range(5):
            seed = random.randint(0, 100000)
            run_experiment(
                example,
                exp_prefix="tf-ddpg-benchmark",
                seed=seed,
                mode='local',
                variant={
                    'env_class': env_class,
                    'version': str(env_class),
                }
            )
