"""
Exampling of running DDPG on HalfCheetah.
"""
import random

from railrl.envs.env_utils import gym_env
from railrl.envs.time_limited_env import TimeLimitedEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from railrl.torch.ddpg import DDPG
from railrl.torch.easy_v_ql import EasyVQFunction
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv

from rllab.envs.normalized_env import normalize


def example(variant):
    # env = HalfCheetahEnv()
    # env = PointEnv()
    env = gym_env("Pendulum-v0")
    # env = HopperEnv()
    horizon = variant['algo_params']['max_path_length']
    env = TimeLimitedEnv(env, horizon)
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            num_epochs=50,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            target_hard_update_period=100,
            batch_size=128,
            max_path_length=1000,
        )
    )
    seed = random.randint(0, 999999)
    run_experiment(
        example,
        exp_prefix="dev-ddpg",
        seed=seed,
        mode='here',
        variant=variant,
        use_gpu=True,
    )
