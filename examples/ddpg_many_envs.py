"""
Run DDPG on many environments
"""
import random

from railrl.algos.ddpg import DDPG
from railrl.launchers.launcher_util import (
    run_experiment,
    get_standard_env_ids,
    get_env_settings,
)
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from rllab.exploration_strategies.ou_strategy import OUStrategy


def example(variant):
    env_settings = get_env_settings(variant['env_id'])
    env = env_settings['env']
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
        n_epochs=100,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    for env_id in get_standard_env_ids():
        for _ in range(3):
            seed = random.randint(0, 100000)
            run_experiment(
                example,
                exp_prefix="3-6-big-benchmark-naf-2",
                seed=seed,
                mode='ec2',
                variant={'env_id': env_id}
            )
