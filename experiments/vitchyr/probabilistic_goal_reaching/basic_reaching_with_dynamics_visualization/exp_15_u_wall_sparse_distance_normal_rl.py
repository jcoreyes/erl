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
            num_epochs=200,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        dynamics_model_version='fixed_standard_gaussian',
        dynamics_model_config=dict(
            hidden_sizes=[64, 64],
            output_activations=['tanh', 'tanh'],
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
            num_columns_per_rollout=7,
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
        eval_env_ids={
            'hard_init': 'Point2D-Easy-UWall-Hard-Init-v2',
            'random_init': 'Point2D-Easy-UWall-v2',
        },
        # env_id='Point2D-FlatWall-v2',
        # eval_env_ids={
        #     'hard_init': 'Point2D-FlatWall-Hard-Init-v2',
        #     'random_init': 'Point2D-FlatWall-v2',
        # },
        env_id='Point2D-FlatWall-v2',
    )

    search_space = {
        'pgr_trainer_kwargs.reward_type': [
            # 'normal',
            'discounted',
            # 'discounted_plus_time_kl',
        ],
        'dynamics_model_version': [
            'fixed_standard_gaussian',
            # 'learned_model',
        ],
        'pgr_trainer_kwargs.discount_type': [
            'prior'
        ],
        'replay_buffer_kwargs': [
            dict(
                fraction_future_context=0.8,
                fraction_distribution_context=0.0,
                fraction_next_context=0.,
                max_size=int(1e6),
            ),
            dict(
                fraction_future_context=0.25,
                fraction_distribution_context=0.25,
                fraction_next_context=0.25,
                max_size=int(1e6),
            ),
            dict(
                fraction_future_context=0.25,
                fraction_distribution_context=0.25,
                fraction_next_context=0.5,
                max_size=int(1e6),
            ),
            dict(
                fraction_future_context=0.25,
                fraction_distribution_context=0.5,
                fraction_next_context=0.25,
                max_size=int(1e6),
            ),
            dict(
                fraction_future_context=0.5,
                fraction_distribution_context=0.25,
                fraction_next_context=0.25,
                max_size=int(1e6),
            ),
        ],
        'env_id': [
            'Point2D-Easy-UWall-Hard-Init-v2',
            # 'Point2D-Easy-UWall-v2',
        ],
        'reward_type': [
            # 'log_prob',
            # 'prob',
            'sparse',
            'negative_distance',
        ],
        'max_path_length': [
            100,
        ],
        'pgr_trainer_kwargs.reward_scale': [
            0.01,
            1.,
        ],
        'action_noise_scale': [
            0.1,
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

    n_seeds = 2
    mode = 'htp'
    exp_name = 'pgr--basic-reaching--exp-15--u-wall-sparse-distance-normal-rl'

    if mode == 'local':
        variant['algo_kwargs'] =dict(
            batch_size=32,
            num_epochs=4,
            num_eval_steps_per_epoch=100,
            num_expl_steps_per_train_loop=100,
            num_trains_per_train_loop=100,
            min_num_steps_before_training=100,
        )
        variant['save_video'] = True

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
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
