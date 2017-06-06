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
from railrl.algos.ddpg import DDPG, TargetUpdateMode
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import GymEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize


def example(variant):
    # env = HalfCheetahEnv()
    # pointenv = PointEnv()
    # env = ProxyEnv(PendulumEnv())
    env = CartpoleEnv()
    env = normalize(env)
    # env = gym_env("Pendulum-v0")
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
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            n_epochs=100,
            eval_samples=100,
            epoch_length=1000,
            batch_size=32,
            # target_update_mode=TargetUpdateMode.SOFT,
            # soft_target_tau=0.001,
            target_update_mode=TargetUpdateMode.HARD,
            hard_update_period=100,
        ),
    )
    for seed in range(8):
        run_experiment(
            example,
            exp_prefix="6-5-ddpg-tf-hard-vs-soft",
            seed=seed,
            mode='ec2',
            variant=variant,
        )
