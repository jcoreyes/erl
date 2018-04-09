"""
Run DQN on grid world.
"""

import gym
import numpy as np

from railrl.torch.dqn.double_dqn import DoubleDQN

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.dqn.dqn import DQN
from railrl.torch.networks import Mlp, CNN
from torch import nn as nn
from railrl.torch.modules import HuberLoss
from railrl.envs.wrappers import DiscretizeEnv, ImageEnv
from railrl.torch.ddpg.ddpg import DDPG
from railrl.envs.mujoco.discrete_reacher import DiscreteReacherEnv

from railrl.launchers.launcher_util import setup_logger

def experiment(variant):
    env = DiscretizeEnv(gym.make(variant['env_id']), variant['bins'])
    training_env = DiscretizeEnv(gym.make(variant['env_id']), variant['bins'])
    env = ImageEnv(env)
    training_env = ImageEnv(env)

    qf = CNN(
        output_size=env.action_space.n,
        **variant['cnn_params']
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
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            batch_size=128,
            max_path_length=200,
            discount=0.99,
            epsilon=0.2,
            tau=0.001,
            hard_update_period=1000,
            replay_buffer_size=10000,
            save_environment=True,  # Can't serialize CartPole for some reason
        ),
        cnn_params=dict(
            input_size=32,
            in_channel=3,
            kernel_sizes=[5, 5],
            n_channels=[8, 16],
            strides=[1, 1],
            pool_sizes=[1, 1],
            paddings=[2, 2],
        ),
        algo_class=DoubleDQN,#DDPG,#DoubleDQN,
        qf_criterion_class=HuberLoss,
        bins=10,
        env_id='InvertedPendulum-v1',
    )
    search_space = {
        'env_id': [
            'InvertedPendulum-v1',
        ],
        'bins': [10],
        'algo_class': [
            DoubleDQN,
            #DQN,
        ],
        # 'algo_params.use_hard_updates': [True, False],
        'qf_criterion_class': [
            #nn.MSELoss,
            HuberLoss,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    #setup_logger('dqn-images-experiment', variant=variant)
    #experiment(variant)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
#        for i in range(3):
        run_experiment(
            experiment,
            variant=variant,
            exp_id=exp_id,
            exp_prefix="dqn-images-inverted-pendulum",
            mode='local',
            # use_gpu=False,
            # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
            # mode='local',
            # use_gpu=True,
        )
