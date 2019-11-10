from railrl.launchers.launcher_util import run_experiment
import railrl.misc.hyperparameter as hyp
from railrl.launchers.experiments.murtaza.rfeatures_rl import state_td3bc_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='SawyerDoorHookResetFreeEnv-v1',
        algo_kwargs=dict(
            num_epochs=500,
            max_path_length=100,
            batch_size=128,
            num_eval_steps_per_epoch=500,
            num_expl_steps_per_train_loop=1000,
            num_trains_per_train_loop=1000,
            min_num_steps_before_training=10000,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            demo_path=None,
            demo_off_policy_path=None,
            bc_num_pretrain_steps=10000,
            q_num_pretrain_steps=10000,
            rl_weight=1.0,
            bc_weight=0,
        ),
        replay_buffer_kwargs=dict(
            max_size=1000000,
            fraction_goals_rollout_goals=0.5,
            fraction_goals_env_goals=0.5,
        ),
        qf_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        policy_kwargs=dict(
            hidden_sizes=[400, 300],
        ),
        save_video=False,
        exploration_noise=.3,
    )

    search_space = {
        'exploration_noise':[.3],


    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_prefix = 'test'

    n_seeds = 1
    mode = 'ec2'
    exp_prefix = 'door_reset_free_state_td3_confirm'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                state_td3bc_experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                num_exps_per_instance=3,
                skip_wait=False,
            )
