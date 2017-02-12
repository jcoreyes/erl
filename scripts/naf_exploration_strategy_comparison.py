from railrl.exploration_strategies.simple_gaussian_strategy import (
    SimpleGaussianStrategy
)
from railrl.launchers.algo_launchers import naf_launcher
from rllab.exploration_strategies.gaussian_strategy import GaussianStrategy
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite


class SimpleGaussianStrategyTmp(SimpleGaussianStrategy):
    def __init__(self, env):
        super().__init__(self, env, sigma=0.15)


def main():
    env_params = dict(
        env_id='cart',
        normalize_env=True,
        gym_name="",
    )
    algo_params = dict(
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
    )
    variants = [
        {
            'es_init': OUStrategy,
            'es_init_params': dict(),
            'env_params': env_params,
            'algo_params': algo_params,
        },
        {
            'es_init': SimpleGaussianStrategy,
            'es_init_params': dict(sigma=0.15),
            'env_params': env_params,
            'algo_params': algo_params,
        },
        {
            'es_init': GaussianStrategy,
            'es_init_params': dict(),
            'env_params': env_params,
            'algo_params': algo_params,
        },
    ]
    for variant in variants:
        for seed in range(3):
            run_experiment_lite(
                naf_launcher,
                n_parallel=1,
                snapshot_mode="last",
                exp_prefix="naf-es-comparison",
                seed=seed,
                use_cloudpickle=True,
                variant=variant,
            )

if __name__ == "__main__":
    main()
