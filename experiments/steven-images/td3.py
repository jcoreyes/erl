from gym.envs.mujoco import HopperEnv
import gym

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.launchers.launcher_util import setup_logger
from rlkit.torch.networks import ConcatMlp, TanhMlpPolicy, CNN, CNNPolicy, MergedCNN
from rlkit.torch.td3.td3 import TD3
from rlkit.torch.ddpg.ddpg import DDPG

import rlkit.images.viewers as viewers
from rlkit.envs.wrappers import ImageEnv
import torch

def experiment(variant):
    imsize = variant['imsize']
    history = variant['history']
    env = gym.make(variant['env_id'])
    env = NormalizedBoxEnv(ImageEnv(env,
                                    imsize=imsize,
                                    keep_prev=history - 1,
                                    init_viewer=variant['init_viewer']))
    es = GaussianStrategy(
        action_space=env.action_space,
    )
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size
    qf1 = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels=history,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])

    qf2 = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels= history,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])

    policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       output_size=action_dim,
                       input_channels= history,
                       **variant['cnn_params'],
                       output_activation=torch.tanh,
    )

    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = TD3(
        env,
        qf1=qf1,
        qf2=qf2,
        policy=policy,
        exploration_policy=exploration_policy,
        policy_and_target_update_period=15,
        policy_learning_rate=1e-5,
        **variant['algo_kwargs']
    )
    """    algorithm = DDPG(
        env,
        qf=qf1,
        policy=policy,
#        qf_weight_decay=.01,
        exploration_policy=exploration_policy,
        **variant['algo_kwargs']
    )"""

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        imsize=16,
        history=3,
        init_viewer=viewers.inverted_pendulum_v2_init_viewer,
        algo_kwargs=dict(
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            max_path_length=200,
            discount=0.99,
            replay_buffer_size=int(1E4),
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[16, 16],
            strides=[2, 2],
            pool_sizes=[1, 1],
            hidden_sizes=[128, 64],
            paddings=[0, 0],
            use_layer_norm=False,
        ),

        env_id='InvertedPendulum-v2',

    )
    setup_logger('name-of-td3-experiment', variant=variant)
    experiment(variant)

    for i in range(2):
        run_experiment(
            experiment,
            variant=variant,
            exp_id=0,
            exp_prefix="TD3-images-inverted-pendulum",
            mode='local',
            # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
            # mode='local',
            #use_gpu=True,
        )
