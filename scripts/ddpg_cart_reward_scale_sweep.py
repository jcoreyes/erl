"""
See how reward scale affects learning.

TODO: DeterministicHyperparameterSweeper doesn't work in stub mode because
**(stub object) just hangs.
different.
"""
from railrl.misc.hyperparameter import DeterministicHyperparameterSweeper
from railrl.misc.scripts_util import timestamp
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic

from railrl.algos.ddpg import DDPG
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.envs.base import TfEnv


def main():
    env = TfEnv(CartpoleEnv())
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    default_ddpg_params = dict(
        batch_size=32,
        n_epochs=10,
        epoch_length=1000,
        eval_samples=1000,
        max_path_length=100,
        min_pool_size=1000,
    )
    sweeper = DeterministicHyperparameterSweeper(
        {'scale_reward': [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]},
    )
    exp_prefix = 'ddpg-cart-reward-scale-sweep-{0}'.format(timestamp())
    for ddpg_params in sweeper.iterate_hyperparameters():
        algorithm = DDPG(
            env,
            es,
            policy,
            qf,
            scale_reward=ddpg_params['scale_reward'],
            **default_ddpg_params,
        )

        for seed in range(3):
            run_experiment_lite(
                algorithm.train(),
                n_parallel=1,
                snapshot_mode="last",
                exp_prefix=exp_prefix,
                seed=seed,
                # mode="local",
                # use_cloudpickle=True,
            )


if __name__ == "__main__":
    stub(globals())
    main()
