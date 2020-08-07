"""
Exampling of running DDPG on Double Pendulum.
"""
from rlkit.envs.env_utils import gym_env
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.qfunctions.nn_qfunction import FeedForwardCritic
from rlkit.tf.ddpg import DDPG
from rlkit.tf.policies.nn_policy import FeedForwardPolicy
from rllab.envs.normalized_env import normalize


def example(*_):
    env = normalize(gym_env('Pendulum-v0'))
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
        n_epochs=30,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="ddpg-normalized-pendulum",
        seed=0,
        mode='here',
    )
