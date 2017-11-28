import random

import numpy as np

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.envs.multitask.reacher_7dof import Reacher7DofXyzGoalState, \
    Reacher7DofAngleGoalState
from railrl.envs.multitask.reacher_env import GoalStateSimpleStateReacherEnv
from railrl.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.state_distance.flat_networks import StructuredQF
from railrl.state_distance.tdm_ddpg import TdmDdpg


def experiment(variant):
    env = variant['env_class']()

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    vectorized = variant['algo_params']['tdm_kwargs']['vectorized']
    qf = StructuredQF(
        observation_dim=obs_dim,
        action_dim=action_dim,
        goal_dim=env.goal_dim,
        output_size=env.goal_dim if vectorized else 1,
        **variant['qf_params']
    )
    policy = FeedForwardPolicy(
        obs_dim=obs_dim + env.goal_dim + 1,
        action_dim=action_dim,
        **variant['policy_params']
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
        **variant['her_replay_buffer_params']
    )
    algorithm = TdmDdpg(
        env,
        qf=qf,
        replay_buffer=replay_buffer,
        policy=policy,
        exploration_policy=exploration_policy,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algorithm.cuda()
    algorithm.train()


if __name__ == "__main__":
    n_seeds = 5
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            base_kwargs=dict(
                num_epochs=500,
                num_steps_per_epoch=1000,
                num_steps_per_eval=1000,
                num_updates_per_env_step=10,
                batch_size=128,
                max_path_length=100,
                discount=1,
            ),
            tdm_kwargs=dict(
                vectorized=True,
            ),
            ddpg_kwargs=dict(
                tau=0.01,
            ),
        ),
        her_replay_buffer_params=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
    )
    search_space = {
        'env_class': [
            Reacher7DofAngleGoalState,
        ]
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        for i in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                experiment,
                seed=seed,
                variant=variant,
                exp_id=exp_id,
                exp_prefix="tdm-ddpg-reacher-7dof-angles",
                mode='ec2',
                use_gpu=False,
                # exp_prefix="dev-tdm-ddpg",
                # mode='local',
                # use_gpu=True,
            )
