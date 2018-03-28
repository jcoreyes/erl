"""
Run DQN on grid world.
"""

import gym
import numpy as np
from torch import nn as nn

import railrl.torch.pytorch_util as ptu
from railrl.envs.mujoco.discrete_reacher import DiscreteReacherEnv
from railrl.envs.mujoco.discrete_swimmer import DiscreteSwimmerEnv
from railrl.launchers.launcher_util import setup_logger, run_experiment
from railrl.torch.dqn.double_dqn import DoubleDQN
from railrl.torch.dqn.dqn import DQN
from railrl.torch.networks import Mlp
from gym.envs.mujoco import HopperEnv
from railrl.envs.wrappers import DiscretizeEnv

import railrl.misc.hyperparameter as hyp


def experiment(variant):
    # env = gym.make('CartPole-v0')
    # training_env = gym.make('CartPole-v0')
    # env = DiscreteReacherEnv(num_bins=5, frame_skip=5)
    # env = DiscreteSwimmerEnv()
    env = variant['env_class'](**variant['env_kwargs'])
    env = DiscretizeEnv(env, variant['num_bins'])

    qf = Mlp(
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
        **variant['qf_kwargs']
    )
    qf_criterion = nn.MSELoss()
    # Use this to switch to DoubleDQN
    # algorithm = DoubleDQN(
    algorithm = variant['algo_class'](
        env,
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
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=20,
            discount=0.99,
            epsilon=0.05,
            tau=0.001,
            hard_update_period=500,
            use_hard_updates=True,
            # save_environment=False,  # Can't serialize CartPole for some reason
        ),
        env_kwargs=dict(
        ),
        # algorithm="Double-DQN",
        algorithm="DQN",
        num_bins=5,
    )
    # setup_logger('name-of-experiment', variant=variant)
    # experiment(variant)
    search_space = {
        # 'algo_kwargs.discount': [0.99, 1],
        # 'algo_kwargs.random_action_prob': [0.05, 0.2],
        'algo_class': [DQN, DoubleDQN],
        'qf_kwargs.hidden_sizes': [[32, 32], [300, 300]],
        'env_class': [HopperEnv],
        'num_bins': [5, 4, 3]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['algo_class'] == DQN:
            variant['algorithm'] = "DQN"
        else:
            variant['algorithm'] = "Double DQN"
        for _ in range(3):
            run_experiment(
                experiment,
                exp_prefix='dqn-vs-finite-dqn-hopper',
                mode='ec2',
                # exp_prefix='dev',
                # mode='local',
                variant=variant,
                exp_id=exp_id,
            )
