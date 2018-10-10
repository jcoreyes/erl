import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.vitchyr.multiworld import (
    relabeling_tsac_experiment,
)
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=1000,
                num_steps_per_epoch=1000,
                num_steps_per_eval=5000,
                max_path_length=500,
                num_updates_per_env_step=1,
                batch_size=128,
                discount=0.99,
                min_num_steps_before_training=1000,
                reward_scale=1,
                render=False,
            ),
            her_kwargs=dict(
                observation_key='state_observation',
                desired_goal_key='state_desired_goal',
            ),
            twin_sac_kwargs=dict(),
        ),
        env_id='SawyerPushAndReachXYEnv-ResetFree-v0',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.5,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        vf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='HER-tSAC',
        version='normal',
        save_video_period=100,
        do_state_exp=True,
        save_video=False,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
    )
    search_space = {
        'env_id': [
            'SawyerPushXYEnv-WithResets-v0',
            'SawyerPushAndReachXYEnv-WithResets-v0',
            'SawyerPushXYEnv-CompleteResetFree-v1',
            'SawyerPushAndReachXYEnv-CompleteResetFree-v0',
        ],
        # 'env_kwargs.num_resets_before_puck_reset': [int(1e6)],
        # 'env_kwargs.num_resets_before_hand_reset': [20, int(1e6)],
        'algo_kwargs.base_kwargs.max_path_length': [100, 500],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 2
    mode = 'sss'
    exp_prefix = 'her-tsac-push-state-sweep'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                relabeling_tsac_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                time_in_mins=23*60,
                snapshot_mode='gap_and_last',
                snapshot_gap=100,
            )
