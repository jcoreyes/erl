"""
Experiments with NAF.
"""
import random

import rlkit.torch.pytorch_util as ptu
from rlkit.exploration_strategies.base import (
    PolicyWrappedWithExplorationStrategy
)
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.naf import NafPolicy, NAF
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import \
    InvertedDoublePendulumEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
import rlkit.misc.hyperparameter as hyp


def experiment(variant):
    env = variant['env_class']()
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    policy = NafPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        **variant['policy_params']
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    algorithm = NAF(
        env,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    num_configurations = 1  # for random mode
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev"
    version = "Dev"
    run_mode = "none"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "benchmark-naf-many-envs"
    # version = "Dev"

    # run_mode = 'grid'
    use_gpu = True
    if mode != "here":
        use_gpu = False

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-3,
            batch_size=128,
            max_path_length=1000,
            discount=0.99,
            policy_learning_rate=1e-4,
        ),
        version="NAF",
    )
    search_space = {
        'env_class': [
            InvertedDoublePendulumEnv,
            SwimmerEnv,
            HalfCheetahEnv,
            HopperEnv,
            AntEnv,
        ],
        'policy_params.use_batchnorm': [False, True],
        'policy_params.hidden_size': [32, 100],
        'policy_params.use_exp_for_diagonal_not_square': [False, True],
        'algo_params.tau': [1e-2, 1e-3],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=use_gpu,
                sync_s3_log=True,
                sync_s3_pkl=True,
                periodic_sync_interval=600,
            )
