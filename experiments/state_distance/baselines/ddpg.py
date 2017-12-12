import random

import railrl.misc.hyperparameter as hyp
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState,
)
from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.algos.ddpg import DDPG


def experiment(variant):
    env = variant['env_class']()
    env = normalize_box(
        env,
        **variant['normalize_params']
    )
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    es = OUStrategy(
        action_space=env.action_space,
        **variant['ou_params']
    )
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        **variant['policy_params']
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
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-ddpg-baseline"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "tdm-half-cheetah-short-epoch-nupo-sweep"

    num_epochs = 100
    num_steps_per_epoch = 1000
    num_steps_per_eval = 10000
    max_path_length = 100

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            num_updates_per_env_step=1,
        ),
        normalize_params=dict(
            obs_mean=None,
            obs_std=None,
        ),
        ou_params=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        qf_params=dict(
            observation_hidden_size=300,
            embedded_hidden_size=300,
        ),
        version="DDPG",
        algorithm="DDPG",
    )
    search_space = {
        'env_class': [
            # Reacher7DofXyzGoalState,
            GoalXVelHalfCheetah,
        ],
        'multitask': [True],
        'algo_params.reward_scale': [
            .1, 1, 10, 100,
        ],
        'algo_params.num_updates_per_env_step': [
            1,
            5,
            25,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                exp_id=exp_id,
                seed=seed,
                mode=mode,
                variant=variant,
                use_gpu=False,
            )
