import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from multiworld.core.flat_goal_env import FlatGoalEnv
from multiworld.core.image_env import ImageEnv
from multiworld.envs.pygame.point2d import Point2DEnv
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.envs.wrappers import NormalizedBoxEnv
from railrl.launchers.launcher_util import run_experiment
from railrl.samplers.data_collector import MdpPathCollector
from railrl.torch.networks import MergedCNN, FlattenMlp
from railrl.torch.sac.policies import MakeDeterministic, TanhGaussianPolicy
from railrl.torch.sac.policies import TanhCNNGaussianPolicy
from railrl.torch.sac.twin_sac import TwinSACTrainer
from railrl.torch.torch_rl_algorithm import TorchBatchRLAlgorithm


def experiment(variant):
    ptu.set_gpu_mode(True, 1)
    env = Point2DEnv(
        **variant['env_kwargs']
    )
    imsize = env.render_size
    env = ImageEnv(env, imsize=env.render_size, transpose=True)
    env = FlatGoalEnv(env)
    env = NormalizedBoxEnv(env)

    action_dim = int(np.prod(env.action_space.shape))

    # qf1 = MergedCNN(
    #     input_width=imsize,
    #     input_height=imsize,
    #     output_size=1,
    #     input_channels=3,
    #     added_fc_input_size=action_dim,
    #     **variant['cnn_params']
    # )
    #
    # qf2 = MergedCNN(
    #     input_width=imsize,
    #     input_height=imsize,
    #     output_size=1,
    #     input_channels=3,
    #     added_fc_input_size=action_dim,
    #     **variant['cnn_params']
    # )
    # target_qf1 = MergedCNN(
    #     input_width=imsize,
    #     input_height=imsize,
    #     output_size=1,
    #     input_channels=3,
    #     added_fc_input_size=action_dim,
    #     **variant['cnn_params']
    # )
    #
    # target_qf2 = MergedCNN(
    #     input_width=imsize,
    #     input_height=imsize,
    #     output_size=1,
    #     input_channels=3,
    #     added_fc_input_size=action_dim,
    #     **variant['cnn_params']
    # )
    action_dim = int(np.prod(env.action_space.shape))
    obs_dim = int(np.prod(env.observation_space.shape))

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )

    # policy = TanhCNNGaussianPolicy(
    policy = TanhGaussianPolicy(
        obs_dim=imsize*imsize*3,
        action_dim=action_dim,
        **variant['policy_kwargs']
        # input_width=imsize,
        # input_height=imsize,
        # output_size=action_dim,
        # input_channels=3,
        # **variant['cnn_params']
    )
    eval_env = expl_env = env

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        **variant['eval_path_collector_kwargs']
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
        **variant['expl_path_collector_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = TwinSACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        data_buffer=replay_buffer,
        **variant['algo_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        env_kwargs=dict(
            fixed_goal=(0, 0),
            images_are_rgb=True,
            render_onscreen=False,
            show_goal=True,
            ball_radius=2,
            render_size=8,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
            target_entropy=-1,
        ),
        algo_kwargs=dict(
            max_path_length=100,
            batch_size=128,
            num_epochs=300,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=10000,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=1000,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=1,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            kernel_sizes=[3],
            n_channels=[32],
            strides=[1],
            hidden_sizes=[32, 32],
            paddings=[0],
        ),
        # replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        replay_buffer_size=int(5E4),
        expl_path_collector_kwargs=dict(
            render=False,
            render_kwargs=dict(
                mode='cv2',
            ),
        ),
        eval_path_collector_kwargs=dict(
            render=True,
            render_kwargs=dict(
                mode='cv2',
            ),
        ),
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'ec2'
    exp_prefix = 'point2d-16x16-img-all-fc-goal00'

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=True,
            )
