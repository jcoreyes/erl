import gym
import numpy as np

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import (
    PretrainedCNN,
    MlpQfWithObsProcessor,
)
from rlkit.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter,
)
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)


def experiment(variant):
    # from softlearning.environments.gym import register_image_reach
    # register_image_reach()
    # env = gym.envs.make(
    #     'Pusher2d-ImageReach-v0',
    # )
    from softlearning.environments.gym.mujoco.image_pusher_2d import (
        ImageForkReacher2dEnv
    )

    env_kwargs = {
        'image_shape': (64, 64, 3),
        'arm_goal_distance_cost_coeff': 1.0,
        'arm_object_distance_cost_coeff': 0.0,
    }

    eval_env = ImageForkReacher2dEnv(**env_kwargs)
    expl_env = ImageForkReacher2dEnv(**env_kwargs)

    input_width, input_height, input_channels = eval_env.image_shape

    action_dim = int(np.prod(eval_env.action_space.shape))
    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=input_width,
        input_height=input_height,
        input_channels=input_channels,
        added_fc_input_size=4,
        output_conv_channels=True,
        output_size=None,
    )
    if variant['shared_qf_conv']:
        qf_cnn = PretrainedCNN(**cnn_params)
        qf1 = MlpQfWithObsProcessor(
            obs_processor=qf_cnn,
            output_size=1,
            input_size=action_dim+qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
        qf2 = MlpQfWithObsProcessor(
            obs_processor=qf_cnn,
            output_size=1,
            input_size=action_dim+qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
        target_qf_cnn = PretrainedCNN(**cnn_params)
        target_qf1 = MlpQfWithObsProcessor(
            obs_processor=target_qf_cnn,
            output_size=1,
            input_size=action_dim+qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
        target_qf2 = MlpQfWithObsProcessor(
            obs_processor=target_qf_cnn,
            output_size=1,
            input_size=action_dim+qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
    else:
        qf1_cnn = PretrainedCNN(**cnn_params)
        cnn_output_dim = qf1_cnn.conv_output_flat_size
        qf1 = MlpQfWithObsProcessor(
            obs_processor=qf1_cnn,
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        qf2 = MlpQfWithObsProcessor(
            obs_processor=PretrainedCNN(**cnn_params),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        target_qf1 = MlpQfWithObsProcessor(
            obs_processor=PretrainedCNN(**cnn_params),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        target_qf2 = MlpQfWithObsProcessor(
            obs_processor=PretrainedCNN(**cnn_params),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
    action_dim = int(np.prod(eval_env.action_space.shape))
    policy_cnn = PretrainedCNN(**cnn_params)
    policy = TanhGaussianPolicyAdapter(
        policy_cnn,
        policy_cnn.conv_output_flat_size,
        action_dim,
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
        **variant['eval_path_collector_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    if variant['collection_mode'] == 'batch':
        expl_path_collector = MdpPathCollector(
            expl_env,
            policy,
            **variant['expl_path_collector_kwargs']
        )
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algo_kwargs']
        )
    elif variant['collection_mode'] == 'online':
        expl_path_collector = MdpStepCollector(
            expl_env,
            policy,
            **variant['expl_path_collector_kwargs']
        )
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=expl_path_collector,
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            **variant['algo_kwargs']
        )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
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
            batch_size=32,
            num_epochs=2000,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=10*100,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=1,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            hidden_sizes=[32, 32],
        ),
        # replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        replay_buffer_size=int(5E4),
        expl_path_collector_kwargs=dict(),
        eval_path_collector_kwargs=dict(),
        shared_qf_conv=False,
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )
    exp_prefix = 'dev-railrl-code-kristian-env-image-reach'

    n_seeds = 3
    # mode = 'ec2'
    exp_prefix = 'railrl-code-kristian-env-image-reach'

    search_space = {
        'shared_qf_conv': [
            True,
            # False,
        ],
        'collection_mode': [
            # 'batch',
            'online',
        ]
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
                gpu_id=0,
            )
