import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import CNNPolicy, MergedCNN
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.envs.wrappers import ImageMujocoWithObsEnv
from rlkit.torch.ddpg.ddpg import DDPG
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy

from rlkit.envs.mujoco.pusher2d import Pusher2DEnv
import rlkit.images.camera as camera
import torch

def experiment(variant):
    imsize = variant['imsize']
    history = variant['history']

    #env = InvertedDoublePendulumEnv()#gym.make(variant['env_id'])
    env = Pusher2DEnv()
    partial_obs_size = env.obs_dim
    env = NormalizedBoxEnv(ImageMujocoWithObsEnv(env,
                                    imsize=imsize,
                                    keep_prev=history-1,
                                    init_camera=variant['init_camera']))
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
                   added_fc_input_size=action_dim + partial_obs_size,
                   **variant['cnn_params'])


    policy = CNNPolicy(input_width=imsize,
                       input_height=imsize,
                       added_fc_input_size=partial_obs_size,
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

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        imsize=16,
        history=1,
        env_id='DoubleInvertedPendulum-v2',
        init_camera=camera.pusher_2d_init_camera,
        algo_params=dict(
            num_epochs=1000,
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
            kernel_sizes=[3, 3],
            n_channels=[16, 16],
            strides=[2, 2],
            pool_sizes=[1, 1],
            hidden_sizes=[400, 300],
            paddings=[0, 0],
            use_batch_norm=False,
        ),

        algo_class=DDPG,
        qf_criterion_class=HuberLoss,
    )
    search_space = {
        # 'algo_params.use_hard_updates': [True, False],
        'qf_criterion_class': [
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
                exp_prefix="DDPG-images-pusher",
                mode='local',
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
                #use_gpu=True,
            )
