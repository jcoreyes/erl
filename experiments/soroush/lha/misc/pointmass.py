import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.masking_launcher import (
    masking_sac_experiment
)
from rlkit.launchers.launcher_util import run_experiment
from experiments.steven.masking.reward_fns import (
    TwoObjectPickAndPlace1DEnvRewardFn
)

if __name__ == "__main__":
    variant = dict(
        train_env_id='TwoObjectPickAndPlace1DEnv-v0',
        eval_env_id='TwoObjectPickAndPlace1DEnv-v0',
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        sac_trainer_kwargs=dict(
            reward_scale=100,
            discount=0.99,
            soft_target_tau=1e-3,
            target_update_period=1,
            use_automatic_entropy_tuning=True,
        ),
        max_path_length=50,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=150,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=500,
            num_trains_per_train_loop=2000,
            min_num_steps_before_training=1000,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='ou',
            exploration_noise=0.2,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.5,
            fraction_distribution_context=0.5,
            max_size=int(1e6),
        ),
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            pad_color=0,
        ),
        exploration_goal_sampling_mode='random',
        evaluation_goal_sampling_mode='random',
        do_masking=True,
    )

    search_space = {
        'exploration_policy_kwargs.exploration_version': ['ou'],
        'do_masking': [True, False],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 2
    mode = 'local'
    exp_name = 'test-masking-no-custom-dims'

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'dev'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                masking_sac_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=True,
                num_exps_per_instance=1,
                gcp_kwargs=dict(
                    zone='us-east1-c',
                    gpu_kwargs=dict(
                        gpu_model='nvidia-tesla-k80',
                        num_gpu=1,
                    )
                ),
                time_in_mins=int(2.5*24*60),
            )
