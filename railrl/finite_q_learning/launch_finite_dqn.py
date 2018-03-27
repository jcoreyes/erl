"""
Try out this finite-horizon q learning.
"""

import gym
from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
import numpy as np

import railrl.torch.pytorch_util as ptu
from railrl.envs.mujoco.discrete_reacher import DiscreteReacherEnv
from railrl.envs.mujoco.discrete_swimmer import DiscreteSwimmerEnv
from railrl.envs.mujoco.reacher_env import ReacherEnv
from railrl.envs.wrappers import DiscretizeEnv
from railrl.finite_q_learning.discrete_q_learning import FiniteDiscreteQLearning
from railrl.launchers.launcher_util import setup_logger, run_experiment
from railrl.torch.networks import Mlp
import railrl.misc.hyperparameter as hyp


def experiment(variant):
    # env = gym.make('CartPole-v0')
    # env = DiscreteReacherEnv()
    env = variant['env_class'](**variant['env_kwargs'])
    env = DiscretizeEnv(env, variant['num_bins'])

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=int(np.prod(env.observation_space.shape)),
        output_size=env.action_space.n,
    )
    algorithm = FiniteDiscreteQLearning(
        env,
        qf,
        **variant['algo_kwargs']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = 'local'
    exp_prefix = 'dev'

    n_seeds = 3
    mode = 'ec2'
    exp_prefix = 'dqn-vs-finite-dqn-pendulums'

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=20,
            discount=1.,
            random_action_prob=0.05,
            # save_environment=False,  # Can't serialize CartPole for some reason
        ),
        env_class=InvertedPendulumEnv,
        env_kwargs=dict(
        ),
        algorithm="Finite-DQN",
        num_bins=5,
    )

    search_space = {
        # 'algo_kwargs.discount': [0.99, 1],
        # 'algo_kwargs.random_action_prob': [0.05, 0.2],
        # 'env_kwargs.frame_skip': [2, 5],
        'env_class': [InvertedPendulumEnv, InvertedDoublePendulumEnv],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
