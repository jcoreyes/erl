import gym
import numpy as np

from railrl.torch.dqn.double_dqn import DoubleDQN

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.dqn.dqn import DQN
from railrl.torch.networks import Mlp, CNN, CNNPolicy, MergedCNN
from torch import nn as nn
from railrl.torch.modules import HuberLoss
from railrl.envs.wrappers import ImageEnv
from railrl.torch.ddpg.ddpg import DDPG
from railrl.envs.mujoco.discrete_reacher import DiscreteReacherEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy

from railrl.launchers.launcher_util import setup_logger
from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.mujoco.reacherv2_edit import ReacherEnv
from railrl.envs.mujoco.idp import InvertedDoublePendulumEnv 
import railrl.images.viewers as viewers
import torch

def experiment(variant):
    imsize = variant['imsize']
    history = variant['history']

    env = InvertedDoublePendulumEnv()#gym.make(variant['env_id'])
    env = NormalizedBoxEnv(ImageEnv(env,
                                    imsize=imsize,
                                    keep_prev=history - 1,
                                    init_viewer=variant['init_viewer']))
#    es = GaussianStrategy(
#        action_space=env.action_space,
#    )
    es = OUStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    qf = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels= history,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])


    policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       output_size=action_dim,
                       input_channels=history,
                       **variant['cnn_params'],
                       output_activation=torch.tanh,
    )


    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf=qf,
        policy=policy,
#        qf_weight_decay=.01,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )

    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        imsize=16,
        history=3,
        env_id='DoubleInvertedPendulum-v2',
        init_viewer=viewers.inverted_double_pendulum_init_viewer,
        algo_params=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            batch_size=64,
            max_path_length=150,
            discount=.99,

            use_soft_update=True,
            tau=1e-3,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,

            save_replay_buffer=False,
            replay_buffer_size=int(1E4),
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[16, 16, 16],
            strides=[2, 2, 1],
            pool_sizes=[1, 1, 1],
            hidden_sizes=[400, 300],
            paddings=[0, 0, 0],
            use_layer_norm=True,
        ),

        algo_class=DDPG,
        qf_criterion_class=HuberLoss,
    )
    search_space = {
        'imsize': [
            32,
        ],
        'env_id': [
            'Reacher-v2',
        ],
        'algo_class': [
            DDPG,
        ],
        # 'algo_params.use_hard_updates': [True, False],
        'qf_criterion_class': [
            HuberLoss,
        ],
    }
    setup_logger('dqn-images-experiment', variant=variant)
    experiment(variant)

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
#        for i in range(2):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="DDPG-images-reacher-OU-grayscale-shaped",
                mode='ec2',
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                #use_gpu=True,
            )
