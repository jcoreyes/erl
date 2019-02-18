import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.vitchyr.multiworld import (
    tdm_twin_sac_experiment,
)
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import IntPiecewiseLinearSchedule

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                max_path_length=100,
                num_updates_per_env_step=1,
                batch_size=10000,
                discount=1,
                min_num_steps_before_training=10000,
                reward_scale=1,
                render=False,
            ),
            tdm_kwargs=dict(
                tau_sample_strategy='uniform',
                dense_rewards=True,
                finite_horizon=True,
                max_tau=24,
            ),
            twin_sac_kwargs=dict(),
        ),
        env_id='HalfCheetah-v2',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.5,
            fraction_goals_env_goals=0.25,
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
        algorithm='TDM-TwinSAC',
        version='normal',
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        vectorized=False,
    )
    search_space = {
        'algo_kwargs.base_kwargs.discount': [
            .99,
        ],
        'algo_kwargs.tdm_kwargs.tau_sample_strategy': [
            'uniform',
            'truncated_geometric',
        ],
        'algo_kwargs.tdm_kwargs.truncated_geom_factor': [
            1,
            # 2.,
            # 3.,
            # 4.,
            5,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'
    base_log_dir = None

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'ddr-scale-rewards-big-batch-size'
    # exp_prefix = 'run2'
    # base_log_dir = 'auto'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        if variant['algo_kwargs']['tdm_kwargs']['tau_sample_strategy'] == 'uniform':
            if variant['algo_kwargs']['tdm_kwargs']['truncated_geom_factor'] == 1:
                continue
        for i in range(n_seeds):
            run_experiment(
                tdm_twin_sac_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                time_in_mins=23*60*3,
                snapshot_mode='gap_and_last',
                snapshot_gap=50,
                base_log_dir=base_log_dir,
                use_gpu=True,
            )
