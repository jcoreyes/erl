"""
Run PyTorch DDPG on HalfCheetah.
"""
import random

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.torch.ddpg.ddpg import DDPG
import rlkit.torch.pytorch_util as ptu

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize


def example(variant):
    env = HalfCheetahEnv()
    if variant['normalize']:
        env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        32,
        32,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        32,
        32,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=99,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        version="PyTorch - bigger networks",
        normalize=False,
        size='32',
    )
    for _ in range(3):
        seed = random.randint(0, 999999)
        run_experiment(
            example,
            exp_prefix="ddpg-half-cheetah-check-clean",
            seed=seed,
            mode='ec2',
            variant=variant,
            use_gpu=False,
        )
