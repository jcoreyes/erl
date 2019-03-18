import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.vitchyr.multiworld import (
    relabeling_tsac_experiment,
)
from railrl.launchers.launcher_util import run_experiment

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        max_path_length=100,
        algo_kwargs=dict(
            batch_size=256,
            num_epochs=200,
            num_eval_paths_per_epoch=5,
            num_train_loops_per_epoch=1,
            num_trains_per_train_loop=1000,
        ),
        twin_sac_trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=1.0,
            policy_update_period=1,
            target_update_period=1000,
            train_policy_with_reparameterization=True,
            policy_lr=3E-4,
            qf_lr=3E-4,
            vf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        env_id='Point2DLargeEnv-offscreen-v0',
        replay_buffer_kwargs=dict(
            max_size=int(1E6),
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0,
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
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
    )
    search_space = {}
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    # n_seeds = 5
    # mode = 'ec2'
    exp_prefix = 'point2d-test'

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
