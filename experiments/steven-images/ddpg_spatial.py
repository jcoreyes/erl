import gym
import numpy as np

from railrl.torch.dqn.double_dqn import DoubleDQN

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.dqn.dqn import DQN
from railrl.torch.networks import Mlp
from torch import nn as nn
from railrl.torch.modules import HuberLoss
from railrl.envs.wrappers import DiscretizeEnv
from railrl.torch.ddpg.feat_point_ddpg import FeatPointDDPG
from railrl.torch.networks import FeatPointMlp
from railrl.envs.mujoco.discrete_reacher import DiscreteReacherEnv
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy, AETanhPolicy
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.envs.mujoco.pusher2d import Pusher2DEnv, RandomGoalPusher2DEnv
from railrl.envs.wrappers import ImageMujocoEnv
#from railrl.images.camera import pusher_2d_init_camera

import railrl.images.camera as camera 
from railrl.data_management.env_replay_buffer import AEEnvReplayBuffer


def experiment(variant):
    feat_points = 16
    latent_obs_dim = feat_points * 2
    imsize = 64
    downsampled_size = 32

    env = gym.make('InvertedPendulum-v2').env#RandomGoalPusher2DEnv()
    env = ImageMujocoEnv(env,
                        init_camera=camera.inverted_pendulum_v2_init_camera,
                         imsize=64,
                         keep_prev=0)
    env = NormalizedBoxEnv(env)
    es = GaussianStrategy(
        action_space=env.action_space,
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    ae = FeatPointMlp(
        input_size=imsize,
        downsample_size=downsampled_size,
        input_channels=1,
        num_feat_points=feat_points
    )
    replay_buffer = AEEnvReplayBuffer(
        int(1e5),
        env,
        imsize=imsize,
        downsampled_size=downsampled_size
    )


    qf = FlattenMlp(
        input_size= latent_obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[128, 128]
    )
    policy = AETanhPolicy(
        input_size=latent_obs_dim,
        ae=ae,
        output_size=action_dim,
        hidden_sizes=[128, 128],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )


    algorithm = FeatPointDDPG(
        env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,

        ae=ae,
        imsize=imsize,
        downsampled_size=downsampled_size,
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
            max_path_length=100,
            discount=.99,

            use_soft_update=True,
            tau=1e-2,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,

            save_replay_buffer=False,
        ),

        qf_criterion_class=nn.MSELoss,
        env_id='Reacher-v2'
    )
    search_space = {
        'env_id': [
            # 'Acrobot-v1',
            #'CartPole-v0',
            'Reacher-v2',
            #'InvertedPendulum-v1',
            # 'CartPole-v1',
            # 'MountainCar-v0',
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
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
#        for i in range(2):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="DDPG-pusher2D",
                mode='local',
                # use_gpu=False,
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                use_gpu=True,
            )
