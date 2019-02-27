"""
Run PyTorch Soft Actor Critic on ImagePusher2dEnv.
"""
import random

import numpy as np
import gym

from railrl.torch.modules import HuberLoss
import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.sac.policies import TanhGaussianPolicy
from railrl.torch.sac.policies import TanhCNNGaussianPolicy
from railrl.torch.sac.twin_sac import TwinSAC

from railrl.torch.networks import Mlp, CNN, CNNPolicy, MergedCNN
from torch import nn as nn
from railrl.torch.networks import FlattenMlp, TanhMlpPolicy

from railrl.envs.mujoco.image_pusher_2d_brandon import ImagePusher2dEnv


def experiment(variant):
    
    imsize = variant['imsize']
    
    env = ImagePusher2dEnv(
        [imsize, imsize, 3],
        ctrl_cost_coeff=10.0,
        arm_object_distance_cost_coeff=1.0,
        goal_object_distance_cost_coeff=1.0)
    
    partial_obs_size = env.obs_dim - imsize * imsize * 3
    print("partial dim was " + str(partial_obs_size))
    env = NormalizedBoxEnv(env)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    
    qf1 = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels=3,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])
    
    qf2 = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels=3,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])
    
    vf  = CNN(input_width=imsize,
               input_height=imsize,
               output_size=1,
               input_channels=3,
               **variant['cnn_params'])
    
    policy = TanhCNNGaussianPolicy(input_width=imsize,
                                   input_height=imsize,
                                   output_size=action_dim,
                                   input_channels=3,
                                   **variant['cnn_params'])
    
    algorithm = TwinSAC(
        env=env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        **variant['algo_params']
    )

    algorithm.to(ptu.device)
    algorithm.train()
    
    
if __name__ == "__main__":
    variant = dict(
        imsize=64,
        algo_params=dict(
            num_epochs=2000,
            num_steps_per_epoch=500,
            num_steps_per_eval=250,
            batch_size=256,
            max_path_length=100,
            discount=.99,

            soft_target_tau=1e-2,
            qf_lr=1e-3,
            vf_lr=1e-3,
            policy_lr=1e-3,

            replay_buffer_size=int(2E5),
        ),
        cnn_params=dict(
            kernel_sizes=[5, 5, 3],
            n_channels=[32, 32, 32],
            strides=[3, 3, 2],
            #pool_sizes=[1, 1, 1],
            hidden_sizes=[400, 300],
            paddings=[0, 0, 0],
            #use_batch_norm=True,
        ),

        qf_criterion_class=HuberLoss,
    )
    
    run_experiment(
        experiment,
        variant=variant,
        #exp_id=0,
        exp_prefix="sac-image-reacher-brandon",
        mode='local',
        # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
        # mode='local',
        # use_gpu=True,
    )