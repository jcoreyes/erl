import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import CNN, MergedCNN
from rlkit.torch.networks.experimental import HuberLoss
from rlkit.envs.wrappers import ImageMujocoEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhCNNGaussianPolicy

from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.envs.mujoco.pusher2d import Pusher2DEnv
import rlkit.images.camera as camera


def experiment(variant):
    imsize = variant['imsize']
    history = variant['history']

    env = Pusher2DEnv()#gym.make(variant['env_id']).env
    env = NormalizedBoxEnv(ImageMujocoEnv(env,
                                    imsize=imsize,
                                    keep_prev=history - 1,
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
                   added_fc_input_size=action_dim,
                   **variant['cnn_params'])

    vf  = CNN(input_width=imsize,
               input_height=imsize,
               output_size=1,
               input_channels=history,
               **variant['cnn_params'])

    policy = TanhCNNGaussianPolicy(input_width=imsize,
                                   input_height=imsize,
                                   output_size=action_dim,
                                   input_channels=history,
                                   **variant['cnn_params'],
    )


    algorithm = SoftActorCritic(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        **variant['algo_params']
    )

    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        imsize=64,
        history=4,
        #env_id='InvertedDoublePendulum-v2',
        init_camera=camera.pusher_2d_init_camera,
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
            pool_sizes=[1, 1, 1],
            hidden_sizes=[400, 300],
            paddings=[0, 0, 0],
            use_batch_norm=True,
        ),

        qf_criterion_class=HuberLoss,
    )
    search_space = {
        'env_id': [
            'InvertedDoublePendulum-v2',
        ],
        # 'algo_params.use_hard_updates': [True, False],
        'qf_criterion_class': [
            HuberLoss,
        ],
    }
#    setup_logger('sac-images-inverted-pendulum', variant=variant)
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
                exp_prefix="sac-images-pusher",
                mode='local',
                # exp_prefix="double-vs-dqn-huber-sweep-cartpole",
                # mode='local',
#                use_gpu=True,
            )
