import gym
import numpy as np

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.obs_dict_replay_buffer import ObsDictReplayBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.samplers.data_collector.path_collector import ObsDictPathCollector
from rlkit.samplers.data_collector.step_collector import ObsDictStepCollector
from rlkit.torch.networks import (
    ConcatMlp, MergedCNN, PretrainedCNN, Flatten,
    MlpQfWithObsProcessor,
)
from rlkit.torch.sac.policies import (
    MakeDeterministic, TanhGaussianPolicy,
    TanhCNNGaussianPolicy,
    TanhGaussianPolicyAdapter,
)
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)
import torchvision.models as models
import torch.nn as nn


def experiment(variant):
    import multiworld
    multiworld.register_all_envs()
    env = gym.make('Image48SawyerReachXYEnv-v1')
    observation_key = 'image_proprio_observation'
    input_width, input_height = env.image_shape

    action_dim = int(np.prod(env.action_space.shape))
    cnn_params = variant['cnn_params']
    cnn_params.update(
        input_width=input_width,
        input_height=input_height,
        input_channels=3,
        added_fc_input_size=3,
        output_conv_channels=True,
        output_size=None,
    )
    if variant['shared_qf_conv']:
        qf_cnn = PretrainedCNN(**cnn_params)
        qf1 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(qf_cnn, Flatten()),
            output_size=1,
            input_size=action_dim+qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
        qf2 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(qf_cnn, Flatten()),
            output_size=1,
            input_size=action_dim+qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
        target_qf_cnn = PretrainedCNN(**cnn_params)
        target_qf1 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(target_qf_cnn, Flatten()),
            output_size=1,
            input_size=action_dim+target_qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
        target_qf2 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(target_qf_cnn, Flatten()),
            output_size=1,
            input_size=action_dim+target_qf_cnn.conv_output_flat_size,
            **variant['qf_kwargs']
        )
    else:
        qf1_cnn = PretrainedCNN(**cnn_params)
        cnn_output_dim = qf1_cnn.conv_output_flat_size
        qf1 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(qf1_cnn, Flatten()),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        qf2 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(PretrainedCNN(**cnn_params), Flatten()),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        target_qf1 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(PretrainedCNN(**cnn_params), Flatten()),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
        target_qf2 = MlpQfWithObsProcessor(
            obs_processor=nn.Sequential(PretrainedCNN(**cnn_params), Flatten()),
            output_size=1,
            input_size=action_dim+cnn_output_dim,
            **variant['qf_kwargs']
        )
    policy_cnn = PretrainedCNN(**cnn_params)
    policy = TanhGaussianPolicyAdapter(
        nn.Sequential(policy_cnn, Flatten()),
        policy_cnn.conv_output_flat_size,
        action_dim,
        **variant['policy_kwargs']
    )
    eval_env = expl_env = env

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = ObsDictPathCollector(
        eval_env,
        eval_policy,
        observation_key=observation_key,
        **variant['eval_path_collector_kwargs']
    )
    replay_buffer = ObsDictReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        observation_key=observation_key,
        **variant['replay_buffer_kwargs'],
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
        expl_path_collector = ObsDictPathCollector(
            expl_env,
            policy,
            observation_key=observation_key,
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
        expl_path_collector = ObsDictStepCollector(
            expl_env,
            policy,
            observation_key=observation_key,
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
            max_path_length=50,
            batch_size=256,
            num_epochs=200,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=500,
            # num_epochs=100,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=1,
            # min_num_steps_before_training=100,
        ),
        cnn_params=dict(
            hidden_sizes=[32, 32],
            model_architecture=models.resnet18,
            model_pretrained=False,
        ),
        replay_buffer_size=int(1E6),
        qf_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        policy_kwargs=dict(
            hidden_sizes=[128, 128],
        ),
        replay_buffer_kwargs=dict(
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.0,
        ),
        expl_path_collector_kwargs=dict(
            # render=False,
            # render_kwargs=dict(
            #     mode='cv2',
            # ),
        ),
        eval_path_collector_kwargs=dict(
            # render=False,
            # render_kwargs=dict(
            #     mode='cv2',
            # ),
        ),
        shared_qf_conv=False,
    )
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 3
    # mode = 'ec2'
    exp_prefix = 'sawyer-reach-xy-48x48-resnet18-random-init'

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
