"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG

from railrl.envs.ros.baxter_env import BaxterEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy


def example(*_):
    env = BaxterEnv(update_hz=20)
    es = OUStrategy(
        max_sigma=0.05,
        min_sigma=0.05,
        env_spec=env.spec,
    )
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
        n_epochs=20,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="4-19-baxter-desired-pos",
        seed=0,
        mode='here',
    )