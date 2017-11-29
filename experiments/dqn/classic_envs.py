"""
Run DQN on grid world.
"""
import random

import gym
import numpy as np

from railrl.envs.wrappers import normalize_box
from railrl.torch.algos.double_dqn import DoubleDQN

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.algos.dqn import DQN
from railrl.torch.networks import Mlp
from torch import nn as nn
from railrl.torch.modules import HuberLoss


def experiment(variant):
    env = gym.make(variant['env_id'])
    training_env = gym.make(variant['env_id'])

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    qf_criterion = variant['qf_criterion_class']()
    algorithm = variant['algo_class'](
        env,
        training_env=training_env,
        qf=qf,
        qf_criterion=qf_criterion,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=500,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            epsilon=0.2,
            tau=0.001,
            hard_update_period=1000,
            save_environment=False,  # Can't serialize CartPole for some reason
        ),
        algo_class=DoubleDQN,
        qf_criterion_class=nn.MSELoss,
    )
    search_space = {
        'env_id': [
            # 'Acrobot-v1',
            'CartPole-v0',
            # 'CartPole-v1',
            # 'MountainCar-v0',
        ],
        'algo_class': [
            DoubleDQN,
            DQN,
        ],
        # 'algo_params.use_hard_updates': [True, False],
        'qf_criterion_class': [
            nn.MSELoss,
            HuberLoss,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(3):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                mode='ec2',
                # use_gpu=False,
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                # use_gpu=True,
            )
