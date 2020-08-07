"""
Run DQN on grid world.
"""
import random

import gym
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.dqn.dqn import DQN
from rlkit.torch.networks import Mlp


def experiment(variant):
    # register_grid_envs()
    # env = gym.make("GridMaze1-v0")
    env = gym.make("Pong-ram-v0")

    qf = Mlp(
        hidden_sizes=[100, 100],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    algorithm = DQN(
        env,
        qf=qf,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=100000,
            num_steps_per_eval=100000,
            batch_size=128,
            max_path_length=10000,
            discount=0.99,
            epsilon=0.2,
            tau=0.001,
        ),
    )
    for _ in range(1):
        seed = random.randint(0, 999999)
        run_experiment(
            experiment,
            exp_prefix="try-dqn-pong-ram-4-long",
            seed=seed,
            variant=variant,
            mode='ec2',
            use_gpu=False,
            # mode='local',
            # use_gpu=True,
        )
