"""
Reference script for the lastest "best results yet" for Reacher7DoF
"""
import argparse

from torch.nn import functional as F

import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.reacher_7dof import Reacher7DofFullGoalState
from railrl.envs.wrappers import convert_gym_space, NormalizedBoxEnv
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.ml_util import ConstantSchedule
from railrl.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy
from railrl.state_distance.networks import (
    FFUniversalPolicy,
    FlatUniversalQfunction,
)
from railrl.state_distance.state_distance_q_learning import (
    HorizonFedStateDistanceQLearning,
)
from railrl.torch.modules import HuberLoss


def experiment(variant):
    env = Reacher7DofFullGoalState()
    env = NormalizedBoxEnv(
        env,
        obs_mean=None,
        obs_std=None,
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
    es = variant['sampler_es_class'](
        action_space=action_space,
        **variant['sampler_es_params']
    )
    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    epoch_discount_schedule = variant['epoch_discount_schedule_class'](
        **variant['epoch_discount_schedule_params']
    )
    algo = variant['algo_class'](
        env,
        qf,
        policy,
        exploration_policy,
        epoch_discount_schedule=epoch_discount_schedule,
        qf_criterion=HuberLoss(),
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
    use_gpu = True
    max_path_length = 300
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=101,
            num_steps_per_epoch=300,
            num_steps_per_eval=3000,
            num_updates_per_env_step=50,
            use_soft_update=True,
            tau=0.001,
            batch_size=500,
            # discount=0.99,
            discount=5,
            sparse_reward=True,
            sample_goals_from='environment',
            sample_discount=True,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            use_new_data=True,
            replay_buffer_size=1000000,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            render=args.render,
            save_replay_buffer=True,
        ),
        # algo_class=StateDistanceQLearning,
        algo_class=HorizonFedStateDistanceQLearning,
        qf_class=FlatUniversalQfunction,
        qf_params=dict(
            hidden_sizes=[100, 100],
            hidden_activation=F.tanh,
            output_activation=F.softplus,
            output_multiplier=-1,
        ),
        policy_class=FFUniversalPolicy,
        policy_params=dict(
            fc1_size=100,
            fc2_size=100,
        ),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.15,
            max_sigma=0.2,
            min_sigma=0.2,
        ),
        epoch_discount_schedule_class=ConstantSchedule,
        epoch_discount_schedule_params=dict(
            value=5,
        ),
    )
    run_experiment(
        experiment,
        # exp_prefix="dev-sdql-reacher7dof-reference",
        exp_prefix="local-sdql-reacher7dof-reference",
        mode="local",
        variant=variant,
        use_gpu=use_gpu,
    )
