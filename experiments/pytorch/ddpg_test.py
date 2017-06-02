"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.envs.point_env import PointEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.ddpg import DDPG
from rllab.envs.gym_env import GymEnv

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from railrl.envs.env_utils import gym_env
from rllab.envs.normalized_env import normalize


def example(*_):
    # env = HalfCheetahEnv()
    # env = PointEnv()
    env = gym_env("Pendulum-v0")
    env = normalize(env)
    es = OUStrategy(env_spec=env.spec)
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        num_epochs=100,
        num_steps_per_epoch=10000,
        batch_size=32,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="dev-5-30-torch-ddpg",
        seed=2,
        mode='here',
    )
