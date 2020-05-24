import railrl.misc.hyperparameter as hyp
from railrl.launchers.contextual.image_based import \
    image_based_goal_conditioned_sac_experiment
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='Point2DLargeEnv-v1',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        cnn_kwargs=dict(
            kernel_sizes=[3, 3, 3],
            n_channels=[8, 16, 32],
            strides=[1, 1, 1],
            paddings=[0, 0, 0],
            pool_type='none',
        ),
        sac_trainer_kwargs=dict(
            reward_scale=1,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
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
            fraction_future_context=0.3,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            pad_color=0,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='occasionally_repeat',
            version_to_kwargs=dict(
                occasionally_repeat=dict(
                    repeat_prob=0.5,
                ),
            ),
        ),
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
        reward_type='state_distance',
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_name = 'dev-{}'.format(
        __file__.replace('/', '-').replace('_', '-').split('.')[0]
    )

    n_seeds = 5
    mode = 'sss'
    # exp_name = 'imaged-based-goal-reaching'
    exp_name = 'purple-reaching-bigger-image'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            # variant['seed'] = seed
            run_experiment(
                image_based_goal_conditioned_sac_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=3,
                # slurm_config_name='gpu_fc',
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(60),
            )
