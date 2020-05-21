import railrl.misc.hyperparameter as hyp
from railrl.launchers.contextual.gym_goal_env import (
    sac_on_gym_goal_env_experiment
)
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='FetchReach-v1',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
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
            num_epochs=500,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.3,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        save_video=True,
        save_video_kwargs=dict(
            rows=1,
            columns=3,
            save_video_period=50,
            pad_color=0,
        ),
        renderer_kwargs=dict(
            img_width=320,
            img_height=320,
            output_img_format='CHW',
        ),
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

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'fetch-reach-example'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for seed in range(n_seeds):
            variant['exp_id'] = exp_id
            variant['seed'] = seed
            run_experiment(
                sac_on_gym_goal_env_experiment,
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
                time_in_mins=int(2.5*24*60),
            )
