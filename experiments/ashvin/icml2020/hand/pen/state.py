from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.ashvin.awr_rl import state_td3bc_experiment

from rlkit.launchers.arglauncher import run_variants

if __name__ == "__main__":
    variant = dict(
        env_id='door-v0',
        algo_kwargs=dict(
            batch_size=1024,
            num_epochs=1000,
            num_eval_steps_per_epoch=1000,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
            max_path_length=50,
        ),
        td3_trainer_kwargs=dict(
            discount=0.99,
        ),
        td3_bc_trainer_kwargs=dict(
            discount=0.99,
            demo_path=None,
            demo_off_policy_path=None,
            bc_num_pretrain_steps=10000,
            q_num_pretrain_steps=10000,
            rl_weight=1.0,
            bc_weight=0,
            reward_scale=1.0,
            target_update_period=2,
            policy_update_period=2,
        ),
        replay_buffer_kwargs=dict(
            max_size=int(1e6),
            # fraction_goals_rollout_goals=0.2,
            # fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        save_video=False,
        exploration_noise=.5,
        td3_bc=True,

        num_exps_per_instance=1,
        region='us-west-2',

        logger_variant=dict(
            tensorboard=True,
        ),
    )

    search_space = {
        'seedid': range(5),
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(state_td3bc_experiment, variants, run_id=0)
