"""
Use EC2 to run DDPG on Cartpole.
"""
from railrl.launchers.algo_launchers import (
    my_ddpg_launcher,
    random_action_launcher,
)
from railrl.launchers.launcher_util import run_experiment


def main():
    variant = dict(
        algo_params=dict(
            batch_size=128,
            n_epochs=50,
            epoch_length=100,
            eval_samples=100,
            discount=0.99,
            qf_learning_rate=1e-2,
            policy_learning_rate=1e-2,
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
        policy_params=dict(
            layer_norm=True,
        ),
        qf_params=dict(
            layer_norm=True,
        ),
    )
    seed = 0
    run_experiment(
        my_ddpg_launcher,
        exp_prefix="dev-layernorm-ddpg",
        seed=seed,
        variant=variant,
        mode="here",
    )


if __name__ == "__main__":
    main()
