import argparse

from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction
from railrl.torch.ddpg import DDPG


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
    exp_prefix = "ddpg-half-cheetah"

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

import argparse
import random

from torch.nn import functional as F

import railrl.torch.pytorch_util as ptu
from railrl.tf.state_distance.her import HER
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.her import HerPolicy, HerQFunction
from railrl.torch.modules import HuberLoss
from railrl.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    env = normalize_box(
        env,
        **variant['normalize_params']
    )

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = variant['qf_class'](
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
        **variant['qf_params']
    )
    policy = variant['policy_class'](
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
        **variant['policy_params']
    )
    qf_criterion = variant['qf_criterion_class'](
        **variant['qf_criterion_params']
    )
    es = variant['es_class'](
        action_space=action_space,
        **variant['es_params']
    )
    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = SplitReplayBuffer(
        HerReplayBuffer(
            env=env,
            **variant['replay_buffer_params'],
        ),
        HerReplayBuffer(
            env=env,
            **variant['replay_buffer_params'],
        ),
        fraction_paths_in_train=0.8,
    )
    algo = HER(
        env,
        qf,
        policy,
        exploration_policy,
        qf_criterion=qf_criterion,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "murtaza-edits-baseline-her"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "her-baseline-save-every-10"

    version = "na"
    snapshot_mode = "last"
    snapshot_gap = 10
    use_gpu = True
    if mode != "local":
        use_gpu = False

    max_path_length = 100
    # noinspection PyTypeChecker
    variant = dict(
        version=version,
        algo_params=dict(
            num_epochs=10,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=1,
            use_soft_update=True,
            tau=0.001,
            batch_size=64,
            discount=0.99,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            render=args.render,
            terminate_when_goal_reached=False,
        ),
        qf_class=HerQFunction,
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.softplus,
        ),
        policy_class=HerPolicy,
        policy_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.relu,
        ),
        replay_buffer_params=dict(
            max_size=200000,
            num_goals_to_sample=4,
            goal_sample_strategy='store',
        ),
        env_params=dict(),
        normalize_params=dict(),
        es_class=OUStrategy,
        es_params=dict(
            theta=0.1,
            max_sigma=0.02,
            min_sigma=0.02,
        ),
        qf_criterion_class=HuberLoss,
        # qf_criterion_class=nn.MSELoss,
        qf_criterion_params=dict(
            # delta=1,
        ),
        exp_prefix=exp_prefix,
    )

    """
    Half Cheetah
    """
    variant['env_class'] = GoalXVelHalfCheetah
    variant['algo_params']['num_updates_per_env_step'] = 5
    variant['algo_params']['reward_scale'] = 10
    variant['algo_params']['num_epochs'] = 201
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
            snapshot_gap=snapshot_gap,
        )

    """
    Reacher
    """
    variant['env_class'] = Reacher7DofXyzGoalState
    variant['algo_params']['num_updates_per_env_step'] = 1
    variant['algo_params']['reward_scale'] = 1
    variant['algo_params']['num_epochs'] = 101
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
            snapshot_gap=snapshot_gap,
        )

    """
    Pusher
    """
    variant['env_class'] = CylinderXYPusher2DEnv
    variant['algo_params']['num_updates_per_env_step'] = 1
    variant['algo_params']['reward_scale'] = 1
    variant['algo_params']['num_epochs'] = 101
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
            snapshot_gap=snapshot_gap,
        )
