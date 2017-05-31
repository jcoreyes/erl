"""
Run DDPG on things.
"""
from gym.envs.classic_control import PendulumEnv
from gym.spaces.box import Box

from railrl.envs.env_utils import gym_env
from railrl.envs.point_env import PointEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.nn_policy import FeedForwardPolicy
from railrl.qfunctions.nn_qfunction import FeedForwardCritic
from railrl.algos.ddpg import DDPG
from rllab.envs.gym_env import GymEnv


def example(*_):
    # env = HalfCheetahEnv()
    pointenv = PointEnv()
    # env = ProxyEnv(PendulumEnv())
    env = gym_env("Pendulum-v0")
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
        # eval_samples=100,
        # epoch_length=1000,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="ddpg-half-cheetah",
        seed=2,
        mode='here',
    )
