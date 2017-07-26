"""
State distance test.
"""
import random

from railrl.algos.state_distance.state_distance_q_learning import DQN, \
    StateDistanceQLearning
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize
import railrl.envs.gridcraft
from railrl.envs.env_utils import gym_env


def example(variant):
    env = gym_env("GridMazeAnyStart1-v0")
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = DQN(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = StateDistanceQLearning(
        env,
        exploration_strategy=es,
        qf=qf,
        **variant['algo_params']
    )
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        version="PyTorch - bigger networks",
    )
    seed = random.randint(0, 999999)
    run_experiment(
        example,
        exp_prefix="ddpg-half-cheetah-pytorch",
        seed=seed,
        mode='here',
        variant=variant,
        use_gpu=True,
    )
