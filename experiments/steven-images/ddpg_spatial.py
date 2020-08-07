import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from torch import nn as nn
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.torch.ddpg.feat_point_ddpg import FeatPointDDPG
from rlkit.torch.networks import FeatPointMlp
from rlkit.torch.networks import ConcatMlp, AETanhPolicy
from rlkit.exploration_strategies.gaussian_strategy import GaussianStrategy
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.envs.wrappers import ImageMujocoWithObsEnv
#from rlkit.images.camera import pusher_2d_init_camera
#from rlkit.envs.mujoco.simple_sawyer import SawyerXYZEnv
from rlkit.envs.mujoco.sawyer_gripper_env import SawyerXYZEnv

import rlkit.images.camera as camera
from rlkit.data_management.env_replay_buffer import AEEnvReplayBuffer


def experiment(variant):
    feat_points = 16
    history = 1
    latent_obs_dim = feat_points * 2 * history
    imsize = 64
    downsampled_size = 32

    env = SawyerXYZEnv()
    extra_fc_size = env.obs_dim
    env = ImageMujocoWithObsEnv(env,
                                imsize=imsize,
                                normalize=True,
                                grayscale=True,
                                keep_prev=history-1,
                                init_camera=camera.sawyer_init_camera)
    """env = ImageMujocoEnv(env,
                        imsize=imsize,
                        keep_prev=history-1,
                        init_camera=camera.sawyer_init_camera)"""

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
        int(1e4),
        env,
        imsize=imsize,
        history_length=history,
        downsampled_size=downsampled_size
    )


    qf = ConcatMlp(
        input_size= latent_obs_dim + extra_fc_size + action_dim,
        output_size=1,
        hidden_sizes=[400, 300]
    )
    policy = AETanhPolicy(
        input_size=latent_obs_dim + extra_fc_size,
        ae=ae,
        env=env,
        history_length=history,
        output_size=action_dim,
        hidden_sizes=[400, 300],
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )


    algorithm = FeatPointDDPG(
        ae,
        history,
        env=env,
        qf=qf,
        policy=policy,
        exploration_policy=exploration_policy,
        replay_buffer=replay_buffer,
        extra_fc_size=extra_fc_size,

        imsize=imsize,
        downsampled_size=downsampled_size,
        **variant['algo_params']
    )

    algorithm.to(ptu.device)
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
#    setup_logger('dqn-images-experiment', variant=variant)
#    experiment(variant)

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
#        for i in range(2):
            run_experiment(
                experiment,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="sawyer-spatial-FREEZE",
                mode='local',
                # use_gpu=False,
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                use_gpu=True,
            )
