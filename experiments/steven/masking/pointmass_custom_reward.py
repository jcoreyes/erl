import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.masking_launcher import (
    masking_sac_experiment
)
from rlkit.launchers.launcher_util import run_experiment
from experiments.steven.masking.reward_fns import (
     PickAndPlace1DEnvObjectOnlyRewardFn
)

if __name__ == "__main__":
    variant = dict(
        train_env_id='TwoObjectPickAndPlaceRandomInit1DEnv-v0',
        eval_env_id='TwoObjectPickAndPlaceRandomInit1DEnv-v0',
        # train_env_id='FourObjectPickAndPlaceRandomInit2DEnv-v0',
        # eval_env_id='FourObjectPickAndPlaceRandomInit2DEnv-v0',
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
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=300,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=1000,
        ),
        exploration_policy_kwargs=dict(
            exploration_version='ou',
            exploration_noise=0.3,
        ),
        replay_buffer_kwargs=dict(
            fraction_future_context=0.4,
            fraction_distribution_context=0.4,
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
        masking_reward_fn=PickAndPlace1DEnvObjectOnlyRewardFn,
        mask_dim=2,
        log_mask_diagnostics=False,
        masking_eval_steps=300,
        rotate_masks_for_eval=True,
        # exploration_policy_path='/home/steven/logs/20-04-27-train-expert-policy-gpu/20-04-27-train-expert-policy-gpu_2020_04_27_00_53_47_id205903--s471039/params.pkl',
    )

    search_space = {
        'exploration_policy_kwargs.exploration_version': ['ou'],
        'exploration_policy_kwargs.exploration_noise': [0.0],
        'do_masking': [True],
        'masking_for_exploration': [True],
        'rotate_masks_for_eval': [True],
        'rotate_masks_for_expl': [True],
        'mask_distribution': ['one_hot_masks']
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 7
    mode = 'local'
    exp_name = 'comp-masking'

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'dev'
    use_gpu = (mode == 'local' or mode == 'here_no_doodad')

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                masking_sac_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                use_gpu=use_gpu,
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
