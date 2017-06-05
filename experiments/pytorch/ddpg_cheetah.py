"""
Run PyTorch DDPG on HalfCheetah.
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
    env = HalfCheetahEnv()
    env = normalize(env)
    es = OUStrategy(env_spec=env.spec)
    algorithm = DDPG(
        env,
        exploration_strategy=es,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    variant = dict(
        algo_params=dict(
            num_epochs=30,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            # target_hard_update_period=10000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
        ),
        version="Torch",
    )
    run_experiment(
        example,
        # exp_prefix="6-5-tf-vs-torch-ddpg-half-cheetah",
        exp_prefix="6-5-torch-bn",
        seed=0,
        mode='here',
        variant=variant,
    )
