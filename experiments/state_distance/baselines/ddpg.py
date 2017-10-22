import random

from railrl.envs.env_utils import gym_env
from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG
import railrl.misc.hyperparameter as hyp

from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.pusher import (
    JointOnlyPusherEnv,
)
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofFullGoalState,
)
from railrl.envs.multitask.pusher2d import HandCylinderXYPusher2DEnv
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
)


def experiment(variant):
    env = gym_env("Reacher-v1")
    # env = variant['env_class']()
    env = normalize_box(
        env,
        **variant['normalize_params']
    )
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
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
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-ddpg-baseline"

    # n_seeds = 5
    # mode = "ec2"
    # exp_prefix = "ddpg-reacher-baseline"

    num_steps_per_iteration = 100
    H = 250
    num_iterations = 1000
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=num_iterations,
            num_steps_per_epoch=num_steps_per_iteration,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=128,
            max_path_length=H,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            num_updates_per_env_step=1,
        ),
        version="DDPG",
        normalize_params=dict(
            obs_mean=None,
            obs_std=None,
        ),
    )
    search_space = {
        'env_class': [
            # JointOnlyPusherEnv,
            # Reacher7DofFullGoalState,
            # GoalStateSimpleStateReacherEnv,
            # MultitaskPusher2DEnv,
            # MultitaskPoint2DEnv,
            "ignored",
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
                exp_id=i,
                seed=seed,
                mode=mode,
                variant=variant,
                use_gpu=False,
                snapshot_mode="gap_and_last",
                snapshot_gap=50,
            )
