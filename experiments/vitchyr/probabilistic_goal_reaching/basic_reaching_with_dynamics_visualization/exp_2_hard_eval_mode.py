import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.vitchyr.probabilistic_goal_reaching.launcher import \
    probabilistic_goal_reaching_experiment
from rlkit.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='Point2DLargeEnv-v1',
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
            # num_epochs=4,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        dynamics_model_version='fixed_standard_gaussian',
        dynamics_model_config=dict(
            hidden_sizes=[64, 64],
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.5,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=20,
            rows=3,
            columns=2,
            subpad_length=1,
            subpad_color=127,
            pad_length=1,
            pad_color=0,
            num_columns_per_rollout=5,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='occasionally_repeat',
            repeat_prob=0.5,
        ),
        video_renderer_kwargs=dict(
            width=24,
            height=24,
            output_image_format='CHW',
        ),
        visualize_dynamics=True,
        learn_discount_model=False,
        dynamics_adam_config=dict(
            lr=1e-2,
        ),
        eval_env_ids={
            'hard_init': 'Point2D-Easy-UWall-Hard-Init-v2',
            'random_init': 'Point2D-Easy-UWall-v2',
        },
    )

    search_space = {
        'pgr_trainer_kwargs.reward_type': [
            # 'normal',
            'discounted',
            'discounted_plus_time_kl',
        ],
        'env_id': [
            'Point2D-Easy-UWall-Hard-Init-v2',
            'Point2D-Easy-UWall-v2',
        ],
        'dynamics_model_version': [
            'fixed_standard_gaussian',
            'learned_model',
        ],
        'pgr_trainer_kwargs.target_entropy': [
            -2,
            -1,
            0,
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
    mode = 'ec2'
    exp_name = 'pgr-with-random-and-hard-init-eval'

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
                time_in_mins=600,
            )
