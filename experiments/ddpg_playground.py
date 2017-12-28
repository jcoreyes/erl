"""
Prototype new ideas for DDPG.
"""
import random

from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv


def example(variant):
    env = variant['env_class'](**variant['env_params'])
    env = normalize_box(env)
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
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    exp_prefix = "dev-cartpole-playground"
    mode = 'local'

    n_seeds = 3
    exp_prefix = "optimize-target-policy-different-envs-sweep-3"
    mode = 'ec2'

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
            residual_gradient_weight=0,
        ),
        qf_params=dict(
            observation_hidden_size=400,
            embedded_hidden_size=300,
        ),
        exp_prefix=exp_prefix,
        env_params=dict(),
    )
    search_space = {
        'env_class': [
            SwimmerEnv,
            HalfCheetahEnv,
            AntEnv,
            HopperEnv,
        ],
        'algo_params.tau': [
            1e-2, 1e-3,
        ],
        'algo_params.optimize_target_policy': [
            True, False,
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
