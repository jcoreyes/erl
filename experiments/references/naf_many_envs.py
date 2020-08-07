"""
Run PyTorch NAF on many envs.
"""
import random

import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.naf import NafPolicy, NAF
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize


def example(variant):
    env = variant['env_class']()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = NafPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        100,
    )
    algorithm = NAF(
        env,
        naf_policy=qf,
        exploration_strategy=es,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            naf_policy_learning_rate=1e-4,
        ),
        version="NAF",
    )
    for env_class in [
        SwimmerEnv,
        HalfCheetahEnv,
        AntEnv,
        HopperEnv,
    ]:
        variant['env_class'] = env_class
        variant['version'] = str(env_class)
        for _ in range(5):
            seed = random.randint(0, 999999)
            run_experiment(
                example,
                exp_prefix="naf-benchmarks-envs-pytorch",
                seed=seed,
                mode='ec2',
                variant=variant,
                use_gpu=False,
            )
