"""
Exampling of running DDPG on HalfCheetah.
"""

from algos.ddpg import DDPG
from policies.nn_policy import FeedForwardPolicy
from qfunctions.nn_qfunction import FeedForwardCritic
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.envs.base import TfEnv


def main():
    stub(globals())

    env = TfEnv(HalfCheetahEnv())
    es = OUStrategy(env_spec=env.spec)
    qf = FeedForwardCritic(
        name_or_scope="critic",
        env_spec=env.spec,
    )
    policy = FeedForwardPolicy(
        name_or_scope="actor",
        env_spec=env.spec,
    )
    algorithm = DDPG(
        env,
        es,
        policy,
        qf,
    )

    run_experiment_lite(
        algorithm.train(),
        n_parallel=1,
        snapshot_mode="last",
        exp_prefix="ddpg-half-cheetah",
        seed=2,
    )

if __name__ == "__main__":
    main()
