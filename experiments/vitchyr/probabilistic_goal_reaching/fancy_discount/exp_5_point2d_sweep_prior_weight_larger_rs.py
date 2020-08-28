import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.launcher import \
    probabilistic_goal_reaching_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        qf_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        policy_kwargs=dict(
            hidden_sizes=[64, 64],
        ),
        pgr_trainer_kwargs=dict(
            reward_scale=1,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        discount_factor=0.99,
        reward_type='log_prob',
        max_path_length=20,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=100,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=100,
        ),
        dynamics_model_version='fixed_standard_gaussian',
        dynamics_model_config=dict(
            hidden_sizes=[64, 64],
            output_activations=['tanh', 'tanh'],
        ),
        dynamics_ensemble_kwargs=dict(
            hidden_sizes=[32, 32],
            num_heads=8,
            # output_activations=['tanh', 'tanh'],
        ),
        discount_model_config=dict(
            hidden_sizes=[64, 64],
            # output_activations=['tanh', 'tanh'],
        ),
        dynamics_delta_model_config=dict(
            outputted_log_std_is_tanh=True,
            log_std_max=2,
            log_std_min=-2,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.25,
            fraction_next_context=0.25,
            fraction_distribution_context=0.25,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=20,
            rows=4,
            columns=1,
            subpad_length=1,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
            num_columns_per_rollout=9,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='epsilon_greedy_and_occasionally_repeat',
            repeat_prob=0.5,
            prob_random_action=0.3,
        ),
        video_renderer_kwargs=dict(
            width=128,
            height=128,
            output_image_format='CHW',
        ),
        visualize_dynamics=True,
        visualize_all_plots=True,
        learn_discount_model=True,
        dynamics_adam_config=dict(
            lr=1e-2,
        ),
        prior_discount_weight_schedule_kwargs=dict(
            version='piecewise_linear',
            x_values=[0, 25, 50, 75, 100],
            y_values=[1, 0, 0, 0, 0],
        ),
        env_id='Point2DLargeEnv-v1',
    )

    search_space = {
        'pgr_trainer_kwargs.reward_type': [
            # 'normal',
            'discounted',
            # 'discounted_plus_time_kl',
        ],
        'dynamics_model_version': [
            # 'learned_model_ensemble',
            # 'learned_model',
            # 'fixed_standard_gaussian',
            # 'learned_model_laplace',
            'fixed_standard_laplace',
        ],
        'pgr_trainer_kwargs.discount_type': [
            # 'prior',
            'computed_from_qr',
        ],
        'pgr_trainer_kwargs.multiply_bootstrap_by_prior_discount': [
            True,
            False,
        ],
        'pgr_trainer_kwargs.auto_init_qf_bias': [
            True,
            False,
        ],
        # 'pgr_trainer_kwargs.initial_weight_on_prior_discount': [
        #     1.0,
        # ],
        # 'pgr_trainer_kwargs.prior_discount_weight_drop_rate': [
        #     .2,
        # ],
        'replay_buffer_kwargs': [
            # dict(
            #     fraction_future_context=0.,
            #     fraction_distribution_context=0.0,
            #     fraction_next_context=0.,
            #     max_size=int(1e6),
            # ),
            dict(
                fraction_future_context=0.8,
                fraction_distribution_context=0.0,
                fraction_next_context=0.,
                max_size=int(1e6),
            ),
            # dict(
            #     fraction_future_context=0.25,
            #     fraction_distribution_context=0.25,
            #     fraction_next_context=0.25,
            #     max_size=int(1e6),
            # ),
            # dict(
            #     fraction_future_context=0.25,
            #     fraction_distribution_context=0.25,
            #     fraction_next_context=0.5,
            #     max_size=int(1e6),
            # ),
            # dict(
            #     fraction_future_context=0.25,
            #     fraction_distribution_context=0.5,
            #     fraction_next_context=0.25,
            #     max_size=int(1e6),
            # ),
            # dict(
            #     fraction_future_context=0.5,
            #     fraction_distribution_context=0.25,
            #     fraction_next_context=0.25,
            #     max_size=int(1e6),
            # ),
        ],
        'prior_discount_weight_schedule_kwargs.y_values': [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0.5, 0, 0],
        ],
        'reward_type': [
            'log_prob',
            # 'prob',
            # 'sparse',
            # 'negative_distance',
        ],
        'max_path_length': [
            50,
        ],
        'pgr_trainer_kwargs.reward_scale': [
            'auto_normalize_by_max_magnitude_times_10',
            'auto_normalize_by_max_magnitude_times_100',
            # 1.,
        ],
        'action_noise_scale': [
            0.1,
            # 0.2,
            # 0.5,
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

    n_seeds = 1
    mode = 'htp'
    exp_name = 'pgr--fancy-discount--exp-5--point2d-sweep-prior-weight-larger-rs'

    if mode == 'local':
        variant['algo_kwargs'] =dict(
            batch_size=32,
            num_epochs=10,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=100,
        )
        variant['save_video'] = True
        variant['save_video_kwargs']['rows'] = 1
        variant['save_video_kwargs']['save_video_period'] = 1

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if (
                variant['reward_type'] == 'sparse'
            and variant['dynamics_model_version'] != 'fixed_standard_laplace'
        ):
            continue
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            # variant['seed'] = seed
            run_experiment(
                probabilistic_goal_reaching_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=False,
                num_exps_per_instance=2,
                slurm_config_name='cpu_co',
                # slurm_config_name='cpu_co',
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=10*60,
            )
