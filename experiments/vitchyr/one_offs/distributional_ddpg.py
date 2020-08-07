"""
Try out distributional DDPG.
"""
import random

from rlkit.sandbox.distributional import DistributionalDDPG, \
    FeedForwardZFunction
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import FeedForwardPolicy
import rlkit.torch.pytorch_util as ptu

from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize


def example(variant):
    env = variant['env_class']()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    zf = FeedForwardZFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
        **variant['zf_params']
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DistributionalDDPG(
        env,
        zf=zf,
        policy=policy,
        exploration_strategy=es,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    num_bins = 10
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=100,
            max_path_length=10,
            use_soft_update=True,
            tau=1e-2,
            batch_size=8,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            num_bins=num_bins,
            returns_min=-10,
            returns_max=10,
        ),
        zf_params=dict(
            num_bins=num_bins,
        ),
        version="DDPG",
    )
    for env_class in [
        # PointEnv,
        SwimmerEnv,
        # HalfCheetahEnv,
        # AntEnv,
        # HopperEnv,
    ]:
        variant['env_class'] = env_class
        variant['version'] = str(env_class)
        for _ in range(5):
            seed = random.randint(0, 999999)
            run_experiment(
                example,
                exp_prefix="dev-distributional-ddpg",
                seed=seed,
                mode='here',
                variant=variant,
                use_gpu=False,
            )
