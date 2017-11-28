import argparse
import random

from railrl.envs.env_utils import gym_env
from railrl.envs.mujoco.pusher2d import RandomGoalPusher2DEnv
from railrl.envs.multitask.her_half_cheetah import HalfCheetah
from railrl.envs.multitask.her_pusher_env import Pusher2DEnv
from railrl.envs.multitask.her_reacher_7dof_env import Reacher7Dof
from railrl.envs.multitask.multitask_env import multitask_to_flat_env
from railrl.envs.wrappers import normalize_box
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.algos.ddpg import DDPG
import railrl.misc.hyperparameter as hyp

from railrl.envs.multitask.point2d import MultitaskPoint2DEnv
from railrl.envs.multitask.pusher import (
    JointOnlyPusherEnv,
)
from railrl.envs.multitask.reacher_7dof import (
    Reacher7DofFullGoalState,
    Reacher7DofMultitaskEnv, Reacher7DofGoalStateEverything,
    Reacher7DofXyzGoalState, Reacher7DofAngleGoalState)
from railrl.envs.multitask.pusher2d import HandCylinderXYPusher2DEnv, \
    MultitaskPusher2DEnv
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
)


def experiment(variant):
    env = variant['env_class']()
    env = normalize_box(
        env,
        **variant['normalize_params']
    )
    # env = multitask_to_flat_env(env)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-state-distance-ddpg-baseline"

    n_seeds = 5
    mode = "ec2"
    exp_prefix = "ddpg-reacher-7dof-angles-only"

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            render=args.render,
            num_epochs=1000,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            use_soft_update=True,
            tau=1e-2,
            batch_size=64,
            max_path_length=100,
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
    )
    search_space = {
        'env_class': [
            # GoalStateSimpleStateReacherEnv,
            # Reacher7DofXyzGoalState,
            # HandCylinderXYPusher2DEnv,
            # Pusher2DEnv,
            # HalfCheetah,
            Reacher7DofAngleGoalState,
            # Reacher7Dof,
            # RandomGoalPusher2DEnv,
            # MultitaskPusher2DEnv,
            # Reacher7DofMultitaskEnv,
        ],
        # 'algo_params.num_updates_per_env_step': [
        #     1, 5,
        # ],
        # 'algo_params.tau': [
        #     1e-2, 1e-3,
        # ],
        'algo_params.reward_scale': [
            10, 1, 0.1,
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
