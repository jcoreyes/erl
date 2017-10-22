import argparse

from torch.nn import functional as F

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
    HorizonFedStateDistanceQLearning)
from railrl.algos.state_distance.vectorized_sdql import VectorizedDeltaTauSdql, \
    VectorizedTauSdql
from railrl.envs.multitask.reacher_env import GoalStateSimpleStateReacherEnv
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.networks.state_distance import (
    FFUniversalPolicy,
    FlatUniversalQfunction,
    GoalConditionedDeltaModel, VectorizedGoalStructuredUniversalQfunction,
    GoalStructuredUniversalQfunction)
from railrl.torch.modules import HuberLoss
from railrl.torch.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy


def experiment(variant):
    env = GoalStateSimpleStateReacherEnv()
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
    policy = FFUniversalPolicy(
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
    algo = variant['algo_class'](
        env,
        qf,
        policy,
        exploration_policy,
        qf_criterion=HuberLoss(),
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


algo_class_to_qf_class = {
    VectorizedTauSdql: VectorizedGoalStructuredUniversalQfunction,
    VectorizedDeltaTauSdql: GoalConditionedDeltaModel,
    HorizonFedStateDistanceQLearning: GoalStructuredUniversalQfunction,
    StateDistanceQLearning: FlatUniversalQfunction,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    use_gpu = True
    max_path_length = 100
    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=101,
            num_steps_per_epoch=100,
            num_steps_per_eval=1000,
            num_updates_per_env_step=5,
            use_soft_update=True,
            tau=0.001,
            batch_size=500,
            discount=0.99,
            sample_goals_from='environment',
            sample_discount=True,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            replay_buffer_size=1000000,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            render=args.render,
            save_replay_buffer=True,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.softplus,
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        normalize_params=dict(
            obs_mean=None,
            obs_std=[0.7, 0.7, 0.7, 0.6, 40, 5],
        ),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.1,
            min_sigma=0.1,
        ),
        algo_class=VectorizedTauSdql,
    )
    algo_class = variant['algo_class']
    variant['qf_class'] = algo_class_to_qf_class[algo_class]
    run_experiment(
        experiment,
        exp_prefix="sdql-example",
        mode="local",
        variant=variant,
        exp_id=0,
        use_gpu=use_gpu,
        snapshot_mode="last",
    )
