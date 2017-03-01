"""
Use EC2 to run DDPG on Cartpole.
"""
from railrl.launchers.algo_launchers import my_ddpg_launcher
from rllab.misc.instrument import run_experiment_lite


def main():
    variant = dict(
        algo_params=dict(
            batch_size=128,
            n_epochs=50,
            epoch_length=1000,
            eval_samples=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            soft_target_tau=0.01,
            replay_pool_size=1000000,
            min_pool_size=256,
            scale_reward=1.0,
            max_path_length=1000,
            qf_weight_decay=0.00,
            n_updates_per_time_step=5,
        ),
        env_params=dict(
            env_id='cart',
            normalize_env=True,
            gym_name="",
        ),
    )
    seed = 0
    run_experiment_lite(
        my_ddpg_launcher,
        exp_prefix="ddpg-cartpole-example",
        n_parallel=1,
        snapshot_mode="last",
        seed=seed,
        mode="ec2",
        variant=variant,
        use_cloudpickle=True,
    )


if __name__ == "__main__":
    main()
