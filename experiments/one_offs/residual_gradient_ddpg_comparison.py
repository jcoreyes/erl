"""
Run PyTorch DDPG on HalfCheetah.
"""
import random

from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu

from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.normalized_env import normalize


def example(variant):
    env = variant['env_class']()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        **variant['qf_params'],
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
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 5
    exp_prefix = "ddpg-residual-gradient-comparison-sweep-params"
    mode = 'here'

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=1000,
            num_steps_per_eval=100,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=100,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            differentiate_through_target=True,
        ),
        version="DDPG",
        env_class=InvertedDoublePendulumEnv,
    )
    search_space = {
        'algo_params.qf_learning_rate': [1e-2, 1e-3, 1e-4],
        'algo_params.policy_learning_rate': [1e-2, 1e-3, 1e-4],
        'qf_params': [
            dict(
                observation_hidden_size=1000,
                embedded_hidden_size=1000,
            ),
            dict(
                observation_hidden_size=400,
                embedded_hidden_size=300,
            ),
        ],
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
