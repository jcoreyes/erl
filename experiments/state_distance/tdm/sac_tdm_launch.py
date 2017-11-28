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
from railrl.sac.policies import TanhGaussianPolicy
from railrl.state_distance.flat_networks import StructuredQF
from railrl.state_distance.tdm_ddpg import TdmDdpg
from railrl.state_distance.tdm_sac import TdmSac
from railrl.torch.networks import FlattenMlp


def experiment(variant):
    env = variant['env_class']()

    obs_dim = int(np.prod(env.observation_space.low.shape))
    action_dim = int(np.prod(env.action_space.low.shape))
    qf = FlattenMlp(
        input_size=obs_dim + action_dim + env.goal_dim + 1,
        output_size=1,
        **variant['qf_params']
    )
    vf = FlattenMlp(
        input_size=obs_dim + env.goal_dim + 1,
        output_size=1,
        **variant['vf_params']
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim + env.goal_dim + 1,
        action_dim=action_dim,
        **variant['policy_params']
    )
    replay_buffer = HerReplayBuffer(
        env=env,
        **variant['her_replay_buffer_params']
    )
    algorithm = TdmSac(
        env=env,
        policy=policy,
        qf=qf,
        vf=vf,
        replay_buffer=replay_buffer,
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
                vectorized=False,
            ),
            sac_kwargs=dict(
                soft_target_tau=0.01,
            ),
        ),
        her_replay_buffer_params=dict(
            max_size=int(1E6),
            num_goals_to_sample=4,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
        ),
        vf_params=dict(
            hidden_sizes=[300, 300],
        ),
        policy_params=dict(
            hidden_sizes=[300, 300],
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
                exp_prefix="sac-ddpg-reacher-7dof-angles",
                mode='ec2',
                use_gpu=False,
                # exp_prefix="dev-tdm-sac",
                # mode='local',
                # use_gpu=True,
            )
