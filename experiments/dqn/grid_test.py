"""
Run DQN on grid world.
"""
import random

import gym
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.gridcraft import register_grid_envs
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.base import Mlp
from railrl.torch.dqn import DQN


def experiment(variant):
    # register_grid_envs()
    # env = gym.make("GridMaze1-v0")
    env = gym.make("CartPole-v0")

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    algorithm = DQN(
        env,
        qf=qf,
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
            num_steps_per_epoch=200,
            num_steps_per_eval=200,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            epsilon=0.2,
            tau=1,
        ),
    )
    seed = random.randint(0, 999999)
    run_experiment(
        experiment,
        exp_prefix="dqn-grid",
        seed=seed,
        mode='local',
        variant=variant,
        use_gpu=True,
    )
