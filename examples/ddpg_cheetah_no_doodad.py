"""
Example of running PyTorch implementation of DDPG on HalfCheetah.
"""
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment, setup_logger
from railrl.torch.networks import FeedForwardQFunction, FeedForwardPolicy, \
    MlpPolicy, FlattenMlp
from railrl.torch.algos.ddpg import DDPG
import railrl.torch.pytorch_util as ptu

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize


def experiment(variant):
    env = HalfCheetahEnv()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FlattenMlp(
        input_size=(
            int(env.observation_space.flat_dim) + int(env.action_space.flat_dim)
        ),
        output_size=1,
        hidden_sizes=[400, 300],
    )
    policy = MlpPolicy(
        input_size=int(env.observation_space.flat_dim),
        output_size=int(env.action_space.flat_dim),
        hidden_sizes=[400, 300],
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
    )
    setup_logger('name-of-experiment', variant=variant)
    experiment(variant)
    # Or if you have doodad installed:
    # run_experiment(
    #     experiment,
    #     exp_prefix="ddpg-half-cheetah-pytorch",
    #     mode='local',
    #     variant=variant,
    #     use_gpu=True,
    # )
    # Or you can use rllab interface
