"""
Run PyTorch Soft Actor Critic on ImagePusher2dEnv.
"""

import numpy as np

from rlkit.torch.networks.experimental import HuberLoss
import rlkit.torch.pytorch_util as ptu
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.sac.policies import TanhCNNGaussianPolicy
from rlkit.torch.sac.sac import SACTrainer

from rlkit.torch.networks import CNN, MergedCNN

from rlkit.envs.mujoco.image_pusher_2d_brandon import ImageForkReacher2dEnv


def experiment(variant):

    imsize = variant['imsize']

    env = ImageForkReacher2dEnv(
        variant["arm_goal_distance_cost_coeff"],
        variant["arm_object_distance_cost_coeff"],
        [imsize, imsize, 3],
        goal_object_distance_cost_coeff=variant["goal_object_distance_cost_coeff"],
        ctrl_cost_coeff=variant["ctrl_cost_coeff"])

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

    algorithm = SACTrainer(
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
        gap_mode=500,
        arm_goal_distance_cost_coeff=1.0,
        arm_object_distance_cost_coeff=0.0,
        goal_object_distance_cost_coeff=0.0,
        ctrl_cost_coeff=0.0,
        algo_params=dict(
            num_epochs=2000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=500,
            batch_size=256,
            max_path_length=100,
            discount=.99,

            soft_target_tau=5e-3,
            qf_lr=3e-4,
            vf_lr=3e-4,
            policy_lr=3e-4,

            replay_buffer_size=int(2e4),
        ),
        cnn_params=dict(
            kernel_sizes=[5, 5, 3],
            n_channels=[32, 32, 32],
            strides=[3, 3, 2],
            # pool_sizes=[1, 1, 1], this param is giving an error?
            hidden_sizes=[400, 300],
            paddings=[0, 0, 0],
            # use_batch_norm=True, this param is giving an error?
        ),

        qf_criterion_class=HuberLoss,
    )

    PARALLEL = 1
    SERIES = 10

    for j in range(SERIES):

        for i in range(PARALLEL):

            run_experiment(
                experiment,
                variant=variant,
                exp_id=i + PARALLEL * j,
                exp_prefix="sac-image-reacher-brandon-softlearning-hyperparameters-{0}".format(i + PARALLEL * j),
                mode='local',
                skip_wait=i != PARALLEL - 1)
