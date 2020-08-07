import random

import numpy as np

import rlkit.misc.hyperparameter as hyp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.her_replay_buffer import HerReplayBuffer
from rlkit.envs.multitask.half_cheetah import GoalXVelHalfCheetah
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.gcm.gcm_ddpg import GcmDdpg
from rlkit.torch.modules import HuberLoss
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.networks import TanhMlpPolicy


def experiment(variant):
    env = NormalizedBoxEnv(variant['env_class']())

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    gcm = FlattenMlp(
        input_size=env.goal_dim + obs_dim + action_dim + 1,
        output_size=env.goal_dim,
        **variant['gcm_kwargs']
    )
    policy = TanhMlpPolicy(
        input_size=obs_dim + env.goal_dim + 1,
        output_size=action_dim,
        **variant['policy_kwargs']
    )
    es = OUStrategy(
        action_space=env.action_space,
        theta=0.1,
        max_sigma=0.1,
        min_sigma=0.1,
    )
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_kwargs']
    )
    gcm_criterion = variant['gcm_criterion_class'](
        **variant['gcm_criterion_kwargs']
    )
    algo_kwargs = variant['algo_kwargs']
    algo_kwargs['base_kwargs']['replay_buffer'] = replay_buffer
    algorithm = GcmDdpg(
        env,
        gcm=gcm,
        policy=policy,
        exploration_policy=exploration_policy,
        gcm_criterion=gcm_criterion,
        **algo_kwargs
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-ddpg-gcm-launch"

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "gcm-ddpg-half-cheetah-sum-of-distances"

    num_epochs = 500
    num_steps_per_epoch = 1000
    num_steps_per_eval = 1000
    max_path_length = 200

    # noinspection PyTypeChecker
    variant = dict(
        algo_kwargs=dict(
            base_kwargs=dict(
                num_epochs=num_epochs,
                num_steps_per_epoch=num_steps_per_epoch,
                num_steps_per_eval=num_steps_per_eval,
                max_path_length=max_path_length,
                num_updates_per_env_step=25,
                batch_size=128,
                discount=1,
            ),
            gcm_kwargs=dict(
                sample_rollout_goals_from='environment',
                sample_train_goals_from='her',
                cycle_taus_for_rollout=True,
                max_tau=10,
            ),
            tau=0.001,
            gcm_learning_rate=1e-3,
            policy_learning_rate=1e-4,
        ),
        her_replay_buffer_kwargs=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        gcm_kwargs=dict(
            hidden_sizes=[100, 100],
        ),
        policy_kwargs=dict(
            hidden_sizes=[100, 100],
        ),
        gcm_criterion_class=HuberLoss,
        gcm_criterion_kwargs=dict(),
        version="DDPG-GCM",
        algorithm="DDPG-GCM",
    )
    search_space = {
        'env_class': [
            # Reacher7DofXyzGoalState,
            GoalXVelHalfCheetah,
        ],
        'algo_kwargs.gcm_kwargs.sample_rollout_goals_from': [
            'environment',
        ],
        'algo_kwargs.gcm_kwargs.max_tau': [
            0,
            5,
            10,
            20,
        ],
        # 'algo_kwargs.tau': [
            # 1e-2,
            # 1e-3,
        # ],
        # 'algo_kwargs.gcm_learning_rate': [
            # 1e-3, 1e-4,
        # ],
        'algo_kwargs.policy_learning_rate': [
            1e-3, 1e-4,
        ],
        'algo_kwargs.base_kwargs.reward_scale': [
            0.1,
            1,
            10,
        ],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            variant['multitask'] = (
                variant['algo_kwargs']['gcm_kwargs'][
                    'sample_rollout_goals_from'
                ] != 'fixed'
            )
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                exp_prefix=exp_prefix,
                mode=mode,
            )
