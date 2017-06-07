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


def example(variant):
    # env = HalfCheetahEnv()
    env = PointEnv()
    # env = gym_env("Pendulum-v0")
    env = normalize(env)
    es = OUStrategy(env_spec=env.spec, max_sima=0.2, min_sigma=None, theta=0.15)
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            num_epochs=10,
            num_steps_per_epoch=1000,
            num_steps_per_eval=100,
            target_hard_update_period=10000,
            batch_size=32,
            max_path_length=100,
        )
    )
    run_experiment(
        example,
        exp_prefix="6-6-dev-pytorch",
        seed=0,
        mode='here',
        variant=variant,
    )
