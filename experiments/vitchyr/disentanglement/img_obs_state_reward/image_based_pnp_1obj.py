import railrl.misc.hyperparameter as hyp
from railrl.launchers.contextual.image_based import \
    image_based_goal_conditioned_sac_experiment
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='OneObjectPickAndPlace2DEnv-v0',
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
            target_entropy=-3,
        ),
        max_path_length=50,
        algo_kwargs=dict(
            batch_size=128,
            num_epochs=500,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
            # num_epochs=5,
            # num_eval_steps_per_epoch=100,
            # num_expl_steps_per_train_loop=100,
            # num_trains_per_train_loop=100,
            # min_num_steps_before_training=100,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.3,
            fraction_distribution_context=0.5,
            max_size=int(2.5e5),
        ),
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=25,
            # save_video_period=1,
            pad_color=255,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='occasionally_repeat',
            # repeat_prob=0.5,
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
        evaluation_goal_sampling_mode='random',
        exploration_goal_sampling_mode='random',
    )

    search_space = {
        'exploration_policy_kwargs.exploration_version': [
            'epsilon_and_occasionally_repeat',
        ],
        'exploration_policy_kwargs.version_to_kwargs.occasionally_repeat.repeat_prob': [
            0.5,
        ],
        'sac_trainer_kwargs.target_entropy': [
            -1,
            1,
        ],
        'exploration_policy_kwargs.version_to_kwargs.epsilon_greedy.prob_random_action': [
            0.0,
            0.2,
        ],
        'replay_buffer_kwargs.fraction_future_context': [
            0.3,
            0.,
        ],
        'cnn_kwargs.hidden_activation': [
            # 'leaky_relu',
            'relu',
        ],
        # 'cnn_kwargs': [
        #     dict(
        #         kernel_sizes=[4, 4, 4],
        #         n_channels=[8, 16, 32],
        #         strides=[1, 1, 1],
        #         paddings=[1, 1, 1],
        #         pool_type='max2d',
        #         pool_sizes=[2, 2, 1],
        #         pool_strides=[2, 2, 1],
        #         pool_paddings=[1, 1, 0],
        #     ),
        #     dict(
        #         kernel_sizes=[4, 4, 4],
        #         n_channels=[8, 16, 32],
        #         strides=[1, 1, 1],
        #         paddings=[1, 1, 1],
        #         pool_type='max2d',
        #         pool_sizes=[2, 2, 1],
        #         pool_strides=[2, 2, 1],
        #         pool_paddings=[1, 1, 0],
        #         hidden_activation='leaky_relu',
        #     ),
        # ],
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
    mode = 'sss'
    exp_name = 'one-obj-img-obs-state-reward-sweep-round3-black-background'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        imsize = variant.pop('imsize')
        variant['env_renderer_kwargs']['img_width'] = imsize
        variant['env_renderer_kwargs']['img_height'] = imsize
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
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(10*60),
            )
