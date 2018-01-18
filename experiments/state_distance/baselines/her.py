import argparse
import random

from torch import nn

from railrl.data_management.normalizer import IdentityNormalizer
from railrl.exploration_strategies.gaussian_and_epislon import \
    GaussianAndEpislonStrategy
from railrl.state_distance.her import HER, HerQFunction, HerPolicy
from torch.nn import functional as F

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from railrl.envs.multitask.pusher2d import CylinderXYPusher2DEnv
from railrl.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState
from railrl.envs.wrappers import convert_gym_space, NormalizedBoxEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy
from railrl.torch.data_management.normalizer import TorchNormalizer
from railrl.torch.modules import HuberLoss


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    # env = NormalizedBoxEnv(env)

    action_space = convert_gym_space(env.action_space)
    obs_space = convert_gym_space(env.observation_space)
    obs_normalizer = variant['normalizer_class'](
        obs_space.flat_dim, **variant['normalizers_params']
    )
    goal_normalizer = variant['normalizer_class'](
        env.goal_dim, **variant['normalizers_params']
    )

    qf = variant['qf_class'](
        env,
        obs_normalizer=obs_normalizer,
        goal_normalizer=goal_normalizer,
        **variant['qf_params']
    )
    policy = variant['policy_class'](
        env,
        obs_normalizer=obs_normalizer,
        goal_normalizer=goal_normalizer,
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
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['replay_buffer_params'],
    )
    algo = HER(
        env,
        qf,
        policy,
        exploration_policy,
        obs_normalizer=obs_normalizer,
        goal_normalizer=goal_normalizer,
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
    exp_prefix = "dev-her-baseline"

    n_seeds = 2
    mode = "ec2"
    exp_prefix = "her-reacher-sweep-layernorm"

    max_path_length = 50
    num_epochs = 200
    num_steps_per_epoch = 16 * max_path_length
    num_steps_per_eval = 16 * max_path_length

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            # num_epochs=200*50,  # One epoch here = one cycle in HER paper
            num_epochs=num_epochs,
            num_steps_per_epoch=num_steps_per_epoch,
            num_steps_per_eval=num_steps_per_eval,
            num_updates_per_epoch=40,
            use_soft_update=True,
            tau=0.05,
            batch_size=128,
            discount=0.98,
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-3,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            render=args.render,
            terminate_when_goal_reached=False,
            collection_mode='batch',
        ),
        qf_class=HerQFunction,
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.relu,
            layer_norm=True,
            layer_norm_kwargs=dict(
                scale=True,
                center=True,
            ),
        ),
        policy_class=HerPolicy,
        policy_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.relu,
            layer_norm=True,
            layer_norm_kwargs=dict(
                scale=True,
                center=True,
            ),
        ),
        replay_buffer_params=dict(
            max_size=int(1e6),
            num_goals_to_sample=4,
        ),
        env_params=dict(),
        normalizers_params=dict(
            eps=1e-1,
            default_clip_range=5,
        ),
        normalizer_class=TorchNormalizer,
        es_class=GaussianAndEpislonStrategy,
        es_params=dict(
            max_sigma=0.1,
            min_sigma=0.1,
            epsilon=0.2,
        ),
        # qf_criterion_class=HuberLoss,
        qf_criterion_class=nn.MSELoss,
        qf_criterion_params=dict(
            # delta=1,
        ),
        exp_prefix=exp_prefix,
        version="HER - Sparse",
        algorithm="HER - DDPG",
    )
    search_space = {
        # 'replay_buffer_params.goal_sample_strategy': [
        #     'online',
        #     'store',
        # ],
        'env_class': [
            Reacher7DofXyzGoalState,
            # CylinderXYPusher2DEnv,
            # GoalXVelHalfCheetah,
        ],
        'algo_params.terminate_when_goal_reached': [
            True,
        ],
        'algo_params.num_updates_per_epoch': [
            400,
        ],
        'normalizer_class': [
            IdentityNormalizer,
            TorchNormalizer,
        ],
        'qf_criterion_class': [
            nn.MSELoss,
            # HuberLoss,
        ],
        'algo_params.batch_size': [
            128,
        ],
        'algo_params.reward_scale': [
            1,
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
                mode=mode,
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
            )
