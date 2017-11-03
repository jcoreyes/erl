import argparse
import random

from torch.nn import functional as F

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.her import HER
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.multitask.pusher2d import HandCylinderXYPusher2DEnv, \
    CylinderXYPusher2DEnv
from railrl.envs.multitask.reacher_7dof import Reacher7DofGoalStateEverything, \
    Reacher7DofXyzGoalState
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.her import HerPolicy, HerQFunction
from railrl.networks.state_distance import (
    FFUniversalPolicy,
)
from railrl.policies.state_distance import TerminalRewardSampleOCPolicy
from railrl.torch.modules import HuberLoss
from railrl.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy
import railrl.misc.hyperparameter as hyp


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
    run_mode = "none"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "her-baseline-shaped-rewards-no-clipping-300-300-right" \
                 "-discount-and-tau"
    run_mode = 'grid'

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
            discount=5,
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
    if run_mode == 'grid':
        search_space = {
            # 'replay_buffer_params.goal_sample_strategy': [
            #     'online',
            #     'store',
            # ],
            'env_class': [
                CylinderXYPusher2DEnv,
                GoalXVelHalfCheetah,
                Reacher7DofXyzGoalState,
            ],
            'algo_params.num_updates_per_env_step': [
                1, 5, 25
            ],
            # 'algo_params.tau': [
            #     1e-2, 1e-3,
            # ],
            'algo_params.scale_reward': [
                10, 1, 0.1,
            ],
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
                    exp_id=0,
                    use_gpu=use_gpu,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                )
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
                use_gpu=use_gpu,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
            )
