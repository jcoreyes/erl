"""
Use EC2 to run DDPG on Cartpole.
"""
from rlkit.launchers.algo_launchers import (
    my_ddpg_launcher,
    random_action_launcher,
)
from rlkit.launchers.launcher_util import run_experiment
from rlkit.misc import hyperparameter as hp


def main():
    num_hyperparameters = 40
    layer_norm = True
    sweeper = hp.RandomHyperparameterSweeper([
        hp.LogFloatParam("qf_learning_rate", 1e-5, 1e-1),
        hp.LogFloatParam("policy_learning_rate", 1e-5, 1e-1),
        hp.LogFloatParam("reward_scale", 10.0, 0.001),
        hp.LogFloatParam("discount", 0.5, 0.99),
    ])
    for seed in range(num_hyperparameters):
        params_dict = sweeper.generate_random_hyperparameters()
        variant = dict(
            algo_params=dict(
                batch_size=128,
                n_epochs=50,
                epoch_length=1000,
                eval_samples=1000,
                replay_pool_size=1000000,
                min_pool_size=256,
                max_path_length=1000,
                qf_weight_decay=0.00,
                n_updates_per_time_step=5,
                soft_target_tau=0.01,
                **params_dict
            ),
            env_params=dict(
                env_id='cart',
                normalize_env=True,
                gym_name="",
            ),
            policy_params=dict(
                layer_norm=layer_norm,
            ),
            qf_params=dict(
                layer_norm=layer_norm,
            ),
        )
        run_experiment(
            my_ddpg_launcher,
            exp_prefix="3-16-cartpole-ddpg-sweep-test",
            seed=seed,
            variant=variant,
            mode="ec2",
        )


if __name__ == "__main__":
    main()
