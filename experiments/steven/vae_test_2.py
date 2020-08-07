import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.disentanglement.contextual_encoder_distance_launcher import (
    encoder_goal_conditioned_sac_experiment
)
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='OneObjectPickAndPlace2DEnv-v0',
        # env_id='TwoObjectPickAndPlaceRandomInit1DEnv-v0',
        disentangled_qf_kwargs=dict(
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
            single_loss_weight=0.5,
            use_automatic_entropy_tuning=True,
        ),
        num_presampled_goals=5000,
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            # num_epochs=10,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.5,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_debug_video=True,
        debug_visualization_kwargs=dict(
            save_period=20,
            initial_save_period=2,
        ),
        save_video=False,
        save_video_kwargs=dict(
            save_video_period=20,
            rows=3,
            columns=2,
            subpad_length=1,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
            num_columns_per_rollout=5,
            horizon=50,
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
        use_image_observations=False,
        env_renderer_kwargs=dict(
            width=12,
            height=12,
            output_image_format='CHW',
        ),
        video_renderer_kwargs=dict(
            width=48,
            height=48,
            output_image_format='CHW',
        ),
        debug_renderer_kwargs=dict(
            width=48,
            height=48,
            output_image_format='CHW',
        ),
        use_separate_encoder_for_policy=True,
        skip_encoder_mlp=False,
        decoder_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        distance_scatterplot_save_period=20,
        distance_scatterplot_initial_save_period=2,
        train_encoder_as_vae=True,
    )

    search_space = {
        'vae_to_sac_loss_scale': [0.1, 1.0, 0.0],
        'reward_type': [
            'encoder_distance',
        ],
        'encoder_kwargs.output_size': [
            8,
        ],
        'max_path_length': [
            20,
        ],
        'encoder_kwargs.hidden_sizes': [
            [64, 64],
            # [64],
            # [64, 64],
        ],
        'replay_buffer_kwargs.fraction_future_context': [
            0.5,
        ],
        'disentangled_qf_kwargs.architecture': [
            # 'single_head_match_many_heads',
            'many_heads',
        ],
        'env_id': [
            # 'OneObjectPickAndPlaceRandomInit2DEnv-v0',
            'OneObjectPickAndPlace2DEnv-v0',
            'OneObjectPickAndPlaceRandomInit2DEnv-v0'
            # 'TwoObjectPickAndPlaceRandomInit1DEnv-v0',
            # 'TwoObjectPickAndPlaceRandomInit2DEnv-v0',
            # 'TwoObjectPickAndPlace2DEnv-v0',
        ],
        'sac_trainer_kwargs.single_loss_weight': [
            1.0,
            # 0.9,
            0.5,
            # 0.1,
            0.0,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'ec2'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 3
    mode = 'ec2'
    exp_name = 'pnp-state-vae-sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            # variant['seed'] = seed
            run_experiment(
                encoder_goal_conditioned_sac_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=(mode == 'local'),
                num_exps_per_instance=1,
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
