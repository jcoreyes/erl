import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.contextual_env_launcher_util import (
    goal_conditioned_sac_experiment, process_args
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        env_id='Point2DLargeEnv-v1',
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
            num_epochs=101,
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
        observation_key='state_observation',
        desired_goal_key='state_desired_goal',
        achieved_goal_key='state_achieved_goal',
        save_video=True,
        save_video_kwargs=dict(
            save_video_period=10,
            pad_color=0,
        ),

        launcher_config=dict(
            unpack_variant=True,
        )
    )

    search_space = {
        "launcher_config.seedid": range(3),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(goal_conditioned_sac_experiment, variants, process_args)
