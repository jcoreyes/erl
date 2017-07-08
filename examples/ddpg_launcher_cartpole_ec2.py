"""
Use EC2 to run DDPG on Cartpole.
"""
from railrl.launchers.algo_launchers import my_ddpg_launcher
from railrl.launchers.launcher_util import run_experiment


def main():
    variant = dict(
        algo_params=dict(
            batch_size=128,
            n_epochs=30,
            epoch_length=1000,
            eval_samples=100,
            discount=0.99,
            qf_learning_rate=1e-3,
            soft_target_tau=0.01,
            replay_pool_size=10000,
            min_pool_size=128,
            scale_reward=1.0,
            max_path_length=100,
            qf_weight_decay=0.00,
            n_updates_per_time_step=1,
        ),
        env_params=dict(
            env_id='cart',
            normalize_env=True,
            gym_name="",
        ),
    )
    seed = 0
    run_experiment(
        my_ddpg_launcher,
        exp_prefix="ddpg-cartpole-example",
        seed=seed,
        variant=variant,
        mode="ec2",
        n_parallel=1,
        snapshot_mode="last",
    )


if __name__ == "__main__":
    main()
