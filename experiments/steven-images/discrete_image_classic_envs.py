"""
Run DQN on grid world.
"""

import gym
import numpy as np

from railrl.torch.dqn.double_dqn import DoubleDQN

import railrl.images.viewers as viewers
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.dqn.dqn import DQN
from railrl.torch.networks import Mlp, CNN
from torch import nn as nn
from railrl.torch.modules import HuberLoss
from railrl.envs.wrappers import DiscretizeEnv, ImageEnv, NormalizedBoxEnv
from railrl.torch.ddpg.ddpg import DDPG
from railrl.envs.mujoco.discrete_reacher import DiscreteReacherEnv
from railrl.envs.mujoco.pusher2d import Pusher2DEnv

from railrl.launchers.launcher_util import setup_logger


def experiment(variant):
    imsize = variant['imsize']
    history = variant['history']

    env = gym.make(variant['env_id'])
    training_env = gym.make(variant['env_id'])

    env=NormalizedBoxEnv(env)
    training_env=NormalizedBoxEnv(training_env)

    env = ImageEnv(env,
                   imsize=imsize,
                   keep_prev=history - 1,
                   init_viewer=variant['init_viewer'])
    training_env = ImageEnv(training_env,
                            imsize=imsize,
                            keep_prev=history - 1,
                            init_viewer=variant['init_viewer'])

    env = DiscretizeEnv(env, variant['bins'])
    training_env = DiscretizeEnv(training_env, variant['bins'])

    qf = CNN(
        output_size=env.action_space.n,
        input_width=imsize,
        input_height=imsize,
        input_channels=3 * history,
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
            num_epochs=400,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            batch_size=64,
            max_path_length=200,
            discount=0.99,
            epsilon=0.2,
            tau=0.001,
            hard_update_period=1000,
            replay_buffer_size=10000,
            learning_rate=1e-3,
            save_environment=True,  # Can't serialize CartPole for some reason
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[16, 16],
            strides=[2, 2],
            pool_sizes=[1, 1],
            paddings=[0, 0],
            hidden_sizes=[64, 64],
            use_layer_norm=False,
        ),
        imsize=16,
        init_viewer=viewers.inverted_pendulum_v2_init_viewer,
        history=3,
        algo_class=DoubleDQN,#DDPG,#DoubleDQN,
        qf_criterion_class=HuberLoss,
        bins=9,
        env_id='InvertedPendulum-v2',
    )
    search_space = {
        'env_id': [
            'InvertedPendulum-v2',
        ],
        'bins': [9],
        'algo_class': [
            DoubleDQN,
        ],
        'qf_criterion_class': [
            HuberLoss,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
#    setup_logger('dqn-images-experiment', variant=variant)
#    experiment(variant)

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(2):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="dqn-images-InvertedPendulum-16x16-9-bins",
                mode='ec2',
              #  use_gpu=True,
            )
