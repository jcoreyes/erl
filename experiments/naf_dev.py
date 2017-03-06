"""
Fiddle with NAF
"""
from railrl.launchers.launcher_util import run_experiment
from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
from railrl.qfunctions.unbiased.unbiased_naf_qfunction import UnbiasedNAF
from railrl.algos.naf import NAF

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.envs.normalized_env import normalize


def example(*_):
    env = HalfCheetahEnv()
    env = normalize(env)
    es = OUStrategy(env_spec=env.spec)
    # naf_qfunction = QuadraticNAF(
    naf_qfunction = UnbiasedNAF(
        name_or_scope="naf_q",
        env_spec=env.spec,
    )
    algorithm = NAF(
        env,
        es,
        naf_qfunction,
        n_epochs=20,
        batch_size=1024,
        epoch_length=1000,
        eval_samples=500,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="dev-naf",
        seed=2,
    )
