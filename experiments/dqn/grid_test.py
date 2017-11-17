"""
Run DQN on grid world.
"""
import random
import numpy as np

import gym

from railrl.envs.gridcraft import register_grid_envs
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.base import Mlp
from railrl.policies.argmax import ArgmaxDiscretePolicy
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
import railrl.torch.pytorch_util as ptu
from railrl.torch.dqn import DQN

from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.normalized_env import normalize


def experiment(variant):
    register_grid_envs()
    env = gym.make("GridMaze1-v0")

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    policy = ArgmaxDiscretePolicy(qf)
    es = EpsilonGreedy(
        action_space=env.action_space,
        prob_random_action=variant['epsilon']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DQN(
        env,
        exploration_policy=exploration_policy,
        qf=qf,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            # use_soft_update=True,
            # tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            # qf_learning_rate=1e-3,
            # policy_learning_rate=1e-4,
        ),
        epsilon=0.2,
        version="PyTorch - bigger networks",
    )
    seed = random.randint(0, 999999)
    run_experiment(
        experiment,
        exp_prefix="dqn-grid",
        seed=seed,
        mode='local',
        variant=variant,
        use_gpu=True,
    )
