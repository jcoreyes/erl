"""
Run PyTorch DDPG on HalfCheetah.
"""
import random

from railrl.envs.env_utils import get_dim
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from railrl.torch.ddpg import DDPG
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu

from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import (
    InvertedDoublePendulumEnv
)
from railrl.envs.mujoco.pusher2d import Pusher2DEnv
from railrl.envs.multitask.reacher_env import GoalStateSimpleStateReacherEnv
from railrl.envs.wrappers import convert_gym_space, NormalizedBoxEnv
from railrl.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)


def example(variant):
    env = variant['env_class']()
    env = NormalizedBoxEnv(env)
    obs_dim = get_dim(env.observation_space)
    action_dim = get_dim(env.action_space)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        obs_dim,
        action_dim,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        obs_dim,
        action_dim,
        400,
        300,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = DDPG(
        env,
        qf,
        policy,
        exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    exp_prefix = "dev-ddpg-rg-weight-env-hp-sweep"
    mode = 'local'

    exp_prefix = "ddpg-rg-weight-env-hp-sweep-2"
    mode = 'ec2'

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=10000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            residual_gradient_weight=0,
        ),
        version="DDPG",
        env_class=InvertedDoublePendulumEnv,
    )
    search_space = {
        'env_class': [
            Pusher2DEnv,
            GoalStateSimpleStateReacherEnv,
            AntEnv,
            HalfCheetahEnv,
            HopperEnv,
            SwimmerEnv,
            InvertedDoublePendulumEnv,
        ],
        'algo_params.residual_gradient_weight': [
            0.5, 0.1, 0
        ],
        'algo_params.tau': [
            1, 1e-2, 1e-4,
        ],
        'qf_params': [
            # dict(
            #     observation_hidden_size=1000,
            #     embedded_hidden_size=1000,
            # ),
            dict(
                observation_hidden_size=400,
                embedded_hidden_size=300,
            ),
        ],
        'algo_params.policy_learning_rate': [1e-3, 1e-4],
        'algo_params.qf_learning_rate': [1e-3, 1e-4],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                example,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )
