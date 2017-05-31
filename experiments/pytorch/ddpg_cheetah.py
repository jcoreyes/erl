"""
Exampling of running DDPG on HalfCheetah.
"""
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.ddpg import DDPG

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv


def example(*_):
    env = HalfCheetahEnv()
    es = OUStrategy(env_spec=env.spec)
    algorithm = DDPG(
        env,
        es,
        num_epochs=100,
        batch_size=1024,
    )
    algorithm.train()


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="dev-5-30-torch-ddpg-half-cheetah",
        seed=2,
        mode='here',
    )
