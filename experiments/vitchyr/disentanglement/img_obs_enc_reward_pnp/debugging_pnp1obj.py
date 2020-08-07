import torch.nn.functional as F

import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.disentanglement.contextual_encoder_distance_launcher import (
    encoder_goal_conditioned_sac_experiment
)
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        # env_id='Point2DLargeEnv-v1',
        env_id='OneObjectPickAndPlace2DEnv-v0',
        disentangled_qf_kwargs=dict(
            encode_state=True,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_using_encoder_settings=dict(
            encode_state=False,
            encode_goal=False,
            detach_encoder_via_goal=False,
            detach_encoder_via_state=False,
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        num_presampled_goals=5000,
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=50,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=100,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.5,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_debug_video=True,
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            # save_video_period=1,
            rows=1,
            columns=2,
            subpad_length=1,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
            num_columns_per_rollout=5,
        ),
        evaluation_goal_sampling_mode='random',
        exploration_goal_sampling_mode='random',
        exploration_policy_kwargs=dict(
            exploration_version='occasionally_repeat',
            repeat_prob=0.5,
        ),
        encoder_cnn_kwargs=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[8, 16, 32],
            strides=[1, 1, 1],
            paddings=[0, 0, 0],
            pool_type='none',
            hidden_activation='relu',
        ),
        use_image_observations=True,
        env_renderer_kwargs=dict(
            width=8,
            height=8,
            output_image_format='CHW',
        ),
        video_renderer_kwargs=dict(
            width=24,
            height=24,
            output_image_format='CHW',
        ),
        debug_renderer_kwargs=dict(
            width=24,
            height=24,
            output_image_format='CHW',
        ),
        use_separate_encoder_for_policy=True,
        skip_encoder_mlp=False,
    )

    search_space = {
        'reward_type': [
            'state_distance',
        ],
        'vectorized_reward': [
            False,
        ],
        'replay_buffer_kwargs.fraction_future_context': [
            # 0.5,
            0.3,
        ],
        'disentangled_qf_kwargs.architecture': [
            'single_head',
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    # n_seeds = 3
    # mode = 'sss'
    # exp_name = 'reach-2d-img-obs-local-is-issue-just-using-encoder'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            # variant['seed'] = seed
            run_experiment(
                encoder_goal_conditioned_sac_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                # slurm_config_name='cpu_co',
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.5*24*60),
            )
