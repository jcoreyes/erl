"""
Fiddle with NAF
"""
import random
from railrl.launchers.launcher_util import (
    run_experiment,
    get_standard_env_ids,
    get_env_settings,
)
from railrl.qfunctions.quadratic_naf_qfunction import QuadraticNAF
from railrl.qfunctions.unbiased.unbiased_naf_qfunction import UnbiasedNAF
from railrl.algos.naf import NAF

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.envs.normalized_env import normalize

from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG


def example(_):
    env_settings = get_env_settings('random2d')
    env = env_settings['env']
    es = OUStrategy(env_spec=env.spec)
    naf_qfunction = QuadraticNAF(
        name_or_scope="naf_q",
        env_spec=env.spec,
    )
    algorithm = NAF(
        env,
        es,
        naf_qfunction,
        n_epochs=25,
        batch_size=1024,
        epoch_length=10000,
        eval_samples=1000,
        replay_pool_size=50000,
    )
    # qf = FeedForwardCritic(
    #     name_or_scope="critic",
    #     env_spec=env.spec,
    # )
    # policy = FeedForwardPolicy(
    #     name_or_scope="actor",
    #     env_spec=env.spec,
    # )
    # algorithm = DDPG(
    #     env,
    #     es,
    #     policy,
    #     qf,
    #     n_epochs=100,
    #     batch_size=1024,
    #     epoch_length=1000,
    #     eval_samples=100,
    #     min_pool_size=1000,
    # )
    algorithm.train()


if __name__ == "__main__":
    seed = random.randint(0, 100000)
    run_experiment(
        example,
        exp_prefix="dev-naf",
        seed=seed,
    )
