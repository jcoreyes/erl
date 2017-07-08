"""
Run PyTorch DDPG on HalfCheetah.
"""
import random

from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
import ipdb

def example(variant):
    env = HalfCheetahEnv()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    use_new_version = variant['use_new_version']
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        use_new_version=use_new_version,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=2,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=1024,
            max_path_length=1000,
        ),
        use_new_version=True,
        version="PyTorch - bigger networks",
    )
    for seed in range(1):
        run_experiment(
            example,
            exp_prefix="ddpg-modified-cheetah-torch-test-DELETE",
            seed=seed,
            mode='here',
            variant=variant,
        )
