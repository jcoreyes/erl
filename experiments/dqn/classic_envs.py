"""
Run DQN on grid world.
"""
import random

import gym
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.base import Mlp
from railrl.torch.dqn import DQN
import railrl.misc.hyperparameter as hyp


def experiment(variant):
    env = gym.make(variant['env_id'])

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
            num_steps_per_epoch=10000,
            num_steps_per_eval=10000,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            epsilon=0.2,
            tau=0.001,
        ),
    )
    search_space = {
        'env_id': [
            'Acrobot-v1',
            'Pendulum-v0',
            'CartPole-v0',
            'CartPole-v1',
            'MountainCar-v0',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(3):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix="dqn-try-many-classic-envs",
                seed=seed,
                variant=variant,
                mode='ec2',
                use_gpu=False,
                # mode='local',
                # use_gpu=True,
            )
