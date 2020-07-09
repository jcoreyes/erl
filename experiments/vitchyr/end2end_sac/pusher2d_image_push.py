import copy

import gym
import numpy as np
import torch.nn as nn

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.launchers.launcher_util import run_experiment
from railrl.pythonplusplus import identity
from railrl.samplers.data_collector import MdpPathCollector
from railrl.samplers.data_collector.step_collector import MdpStepCollector
from railrl.torch.networks import (
    CNN,
    MlpQfWithObsProcessor,
    Split,
    FlattenEach,
    ConcatTuple,
)
from railrl.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicyAdapter,
)
from railrl.torch.sac.sac import SACTrainer
from railrl.torch.torch_rl_algorithm import (
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
        'image_shape': (32, 32, 3),
        'arm_goal_distance_cost_coeff': 0.0,
        'arm_object_distance_cost_coeff': 1.0,
        'goal': (0, -1),

    }

    eval_env = ImageForkReacher2dEnv(**env_kwargs)
    expl_env = ImageForkReacher2dEnv(**env_kwargs)

    input_width, input_height, input_channels = eval_env.image_shape
    image_dim = input_width * input_height * input_channels

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
    non_image_dim = int(np.prod(eval_env.observation_space.shape)) - image_dim
    if variant['shared_qf_conv']:
        qf_cnn = CNN(**cnn_params)
        qf_obs_processor = nn.Sequential(
            Split(qf_cnn, identity, image_dim),
            FlattenEach(),
            ConcatTuple(),
        )

        qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
        qf_kwargs['obs_processor'] = qf_obs_processor
        qf_kwargs['output_size'] = 1
        qf_kwargs['input_size'] = (
                action_dim + qf_cnn.conv_output_flat_size + non_image_dim
        )
        qf1 = MlpQfWithObsProcessor(**qf_kwargs)
        qf2 = MlpQfWithObsProcessor(**qf_kwargs)

        target_qf_cnn = CNN(**cnn_params)
        target_qf_obs_processor = nn.Sequential(
            Split(target_qf_cnn, identity, image_dim),
            FlattenEach(),
            ConcatTuple(),
        )
        target_qf_kwargs = copy.deepcopy(variant['qf_kwargs'])
        target_qf_kwargs['obs_processor'] = target_qf_obs_processor
        target_qf_kwargs['output_size'] = 1
        target_qf_kwargs['input_size'] = (
                action_dim + target_qf_cnn.conv_output_flat_size + non_image_dim
        )
        target_qf1 = MlpQfWithObsProcessor(**target_qf_kwargs)
        target_qf2 = MlpQfWithObsProcessor(**target_qf_kwargs)
    else:
        qf1_cnn = CNN(**cnn_params)
        cnn_output_dim = qf1_cnn.conv_output_flat_size
        qf1 = MlpQfWithObsProcessor(
            obs_processor=qf1_cnn,
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        qf2 = MlpQfWithObsProcessor(
            obs_processor=CNN(**cnn_params),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        target_qf1 = MlpQfWithObsProcessor(
            obs_processor=CNN(**cnn_params),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        target_qf2 = MlpQfWithObsProcessor(
            obs_processor=CNN(**cnn_params),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
    action_dim = int(np.prod(eval_env.action_space.shape))
    policy_cnn = CNN(**cnn_params)
    policy_obs_processor = nn.Sequential(
        Split(policy_cnn, identity, image_dim),
        FlattenEach(),
        ConcatTuple(),
    )
    policy = TanhGaussianPolicyAdapter(
        policy_obs_processor,
        policy_cnn.conv_output_flat_size + non_image_dim,
        action_dim,
        **variant['policy_kwargs']
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
        ),
        algo_kwargs=dict(
            batch_size=256,
            max_path_length=1000,
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10*1000,
            # max_path_length=10,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            kernel_sizes=[3, 3],
            n_channels=[4, 4],
            strides=[1, 1],
            hidden_sizes=[32, 32],
            paddings=[1, 1],
            pool_type='max2d',
            pool_sizes=[2, 2],
            pool_strides=[2, 2],
            pool_paddings=[0, 0],
        ),
        # replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        policy_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        replay_buffer_size=int(1E6),
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

    n_seeds = 5
    # mode = 'ec2'
    exp_prefix = 'railrl-code-sl-image-pusher-prop'

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
                gpu_id=1,
            )
