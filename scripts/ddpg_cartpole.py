"""
Run DDPG on Cartpole.

After testing different settings, here are some timing results:
Setting 1
default_ddpg_params = dict(
    batch_size=32,
    n_epochs=10,
    epoch_length=100,
    eval_samples=100,
    max_path_length=100,
    min_pool_size=100,
With GPU: 35s
Without GPU: 27s


Setting 2
default_ddpg_params = dict(
    batch_size=128,
    n_epochs=10,
    epoch_length=1000,
    eval_samples=1000,
    max_path_length=100,
    min_pool_size=100,
)
With GPU: 1.22s
Without GPU: 43s
"""
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
        batch_size=128,
        n_epochs=10,
        epoch_length=1000,
        eval_samples=1000,
        max_path_length=100,
        min_pool_size=100,
    )
    exp_prefix = 'ddpg-cartpole-speed-{0}'.format(timestamp())
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
        **default_ddpg_params,
    )

    run_experiment_lite(
        algorithm.train(),
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix=exp_prefix,
        seed=1,
    )


if __name__ == "__main__":
    stub(globals())
    main()
