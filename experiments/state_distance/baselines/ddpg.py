import random

import railrl.misc.hyperparameter as hyp
from railrl.envs.multitask.ant_env import GoalXYPosAnt
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah, \
    GoalXPosHalfCheetah
from railrl.envs.multitask.multitask_env import MultitaskToFlatEnv
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.pusher3d import MultitaskPusher3DEnv
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofXyzGoalState,
)
from railrl.envs.multitask.walker2d_env import Walker2DTargetXPos
from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.torch.algos.ddpg import DDPG
from railrl.torch.networks import TanhMlpPolicy, FlattenMlp


def experiment(variant):
    env = variant['env_class'](**variant['env_kwargs'])
    env = normalize_box(
        env,
        **variant['normalize_kwargs']
    )
    if variant['multitask']:
        env = MultitaskToFlatEnv(env)
    es = OUStrategy(
        action_space=env.action_space,
        **variant['ou_kwargs']
    )
    obs_dim = int(env.observation_space.flat_dim)
    action_dim = int(env.action_space.flat_dim)
    qf = FlattenMlp(
        input_size=obs_dim+action_dim,
        output_size=1,
        **variant['qf_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim,
        output_size=action_dim,
        **variant['policy_kwargs']
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
        **variant['algo_kwargs']
    )
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-ddpg-baseline"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "why-are-pusher3d-ddpg-results-not-equivalent"

    num_epochs = 1000
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 250

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            max_path_length=max_path_length,
            use_soft_update=True,
            tau=1e-3,
            batch_size=128,
            discount=0.98,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            num_updates_per_env_step=1,
        ),
        normalize_kwargs=dict(
            obs_mean=None,
            obs_std=None,
        ),
        ou_kwargs=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        policy_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        qf_kwargs=dict(
            hidden_sizes=[300, 300],
        ),
        version="DDPG-no-shaping",
        algorithm="DDPG",
    )
    search_space = {
        'env_class': [
            # Reacher7DofXyzGoalState,
            # GoalXVelHalfCheetah,
            # GoalXYPosAnt,
            # CylinderXYPusher2DEnv,
            # GoalXPosHalfCheetah,
            MultitaskPusher3DEnv,
            # Walker2DTargetXPos,
        ],
        'multitask': [True],
        'env_kwargs': [
            dict(
                reward_coefs=(1, 0, 0),
            ),
            # dict(max_distance=2),
            # dict(max_distance=4),
            # dict(max_distance=6),
            # dict(max_distance=8),
            # dict(max_distance=10),
        ],
        'algo_kwargs.reward_scale': [
            0.1, 1, 10
        ],
        'algo_kwargs.num_updates_per_env_step': [
            1,
        ],
        'algo_kwargs.batch_size': [
            128,
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
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
            )
