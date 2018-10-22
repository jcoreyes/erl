import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import CNNPolicy, MergedCNN
from railrl.torch.modules import HuberLoss
from railrl.torch.ddpg.ddpg import DDPG
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
import torch
from sawyer_control.sawyer_reaching import SawyerXYZReachingEnv
from sawyer_control.sawyer_image import ImageSawyerEnv

import numpy as np

def experiment(variant):

    imsize = variant['imsize']
    history = variant['history']
    env_params = variant['env_params']
    env = SawyerXYZReachingEnv(**env_params)
    env = NormalizedBoxEnv(ImageSawyerEnv(env,
                                    imsize=imsize,
                                    keep_prev=history-1,))

    es = OUStrategy(action_space=env.action_space)
    obs_dim = env.observation_space.low.size
    action_dim = env.action_space.low.size

    qf = MergedCNN(input_width=imsize,
                   input_height=imsize,
                   output_size=1,
                   input_channels= 3 * history,
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])


    policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       output_size=action_dim,
                       input_channels=3 * history,
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
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()

    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        imsize=84,
        history=1,
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            batch_size=64,
            max_path_length=100,
            discount=.99,

            use_soft_update=True,
            tau=1e-3,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,

            save_replay_buffer=False,
            replay_buffer_size=int(2E4),
        ),
        cnn_params=dict(
            kernel_sizes=[5, 5, 5],
            n_channels=[32, 32, 32],
            strides=[3, 3, 2],
            pool_sizes=[1, 1, 1],
            hidden_sizes=[128, 128],
            paddings=[0, 0, 0],
            use_batch_norm=False,
        ),
        env_params=dict(
            action_mode='position',
        ),

        algo_class=DDPG,
        qf_criterion_class=HuberLoss,
    )
    search_space = {
        'qf_criterion_class': [
            HuberLoss,
        ],
    }

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="test",
                mode='here_no_doodad',
                use_gpu=True,
            )
