"""
Fiddle with NAF
"""
from railrl.launchers.launcher_util import run_experiment_here
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.naf_qfunction import NAFQFunction
from railrl.algos.naf import NAF

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy


def example(*_):
    env = HalfCheetahEnv()
    es = OUStrategy(env_spec=env.spec)
    naf_qfunction = NAFQFunction(
        name_or_scope="naf_q",
        env_spec=env.spec,
    )
    algorithm = NAF(
        env,
        env,
        es,
        naf_qfunction,
        n_epochs=100,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment_here(
        example,
        exp_prefix="dev-naf",
        seed=2,
    )
