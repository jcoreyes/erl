from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.launchers.experiments.murtaza.rfeatures_rl import state_td3bc_experiment

if __name__ == "__main__":
    variant = dict(
        env_id='SawyerPushNIPSEasy-v0',
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
            fraction_goals_rollout_goals=0.2,
            fraction_goals_env_goals=0.5,
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
    )

    search_space = {
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # n_seeds = 1
    # mode = 'local'
    # exp_name = 'test'

    n_seeds = 2
    mode = 'gcp'
    exp_name = 'pusher_state_td3_confirm'

    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                state_td3bc_experiment,
                exp_name=exp_name,
                mode=mode,
                variant=variant,
                num_exps_per_instance=3,
                skip_wait=False,
                gcp_kwargs=dict(
                    preemptible=False,
                )
            )
