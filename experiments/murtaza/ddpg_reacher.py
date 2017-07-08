"""
Exampling of running DDPG on Double Pendulum.
"""
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG
from railrl.envs.env_utils import gym_env

def example(*_):
    env = gym_env('Reacher-v1')
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
        n_epochs=50,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="ddpg-reacher",
        seed=0,
        mode='here',
    )
