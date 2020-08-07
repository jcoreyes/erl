import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.vitchyr.multiworld import (
    her_td3_experiment,
    tdm_td3_experiment,
)
from rlkit.launchers.launcher_util import run_experiment

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
                discount=1,
                min_num_steps_before_training=1000,
                reward_scale=100,
                render=False,
            ),
            tdm_kwargs=dict(),
            td3_kwargs=dict(),
        ),
        env_id='SawyerPushAndReachXYEnv-ResetFree-v0',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_are_rollout_goals=0.2,
            fraction_resampled_goals_are_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        algorithm='TDM-TD3',
        version='normal',
        es_kwargs=dict(
            max_sigma=.8,
        ),
        exploration_type='ou',
        save_video_period=100,
        do_state_exp=True,
        imsize=48,
        save_video=False,
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        vectorized=False,
    )
    search_space = {
        'env_id': [
            'SawyerPushXYEnv-WithResets-v0',
            'SawyerPushAndReachXYEnv-WithResets-v0',
        ],
        # 'algo_kwargs.tdm_kwargs.max_tau': [
        #     100, 50, 30,
        # ],
        'algo_kwargs.base_kwargs.reward_scale': [
            100,
        ],
        'algo_kwargs.tdm_kwargs.dense_rewards': [
            True,
        ],
        'algo_kwargs.tdm_kwargs.finite_horizon': [
            False,
        ],
        'algo_kwargs.base_kwargs.discount': [
            0.99,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 3
    # mode = 'ec2'
    # exp_prefix = 'push-tdm-code-non-tdm-settings-again-no-crash-hopefully'

    n_seeds = 3
    mode = 'sss'
    exp_prefix = 'sss-push-tdm-code-non-tdm-settings-again-no-crash-hopefully' \
                 '-with-matplotlib-set'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            run_experiment(
                tdm_td3_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                time_in_mins=20*60,
                snapshot_mode='last',
                # snapshot_gap=100,
            )
