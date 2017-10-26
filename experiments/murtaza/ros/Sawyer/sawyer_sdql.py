import argparse
from torch.nn import functional as F
import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
    HorizonFedStateDistanceQLearning)
from railrl.algos.state_distance.vectorized_sdql import VectorizedDeltaTauSdql, \
    VectorizedTauSdql
from railrl.data_management.her_replay_buffer import HerReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.sawyer_env import MultiTaskSawyerEnv
from railrl.envs.wrappers import convert_gym_space
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
import random

def experiment(variant):
    env = MultiTaskSawyerEnv(**variant['env_params'])
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
    if variant['algo_params']['sample_train_goals_from'] == 'her':
        replay_buffer = SplitReplayBuffer(
            HerReplayBuffer(
                env=env,
                **variant['her_replay_buffer_params']
            ),
            HerReplayBuffer(
                env=env,
                **variant['her_replay_buffer_params']
            ),
            fraction_paths_in_train=0.8,
        )
    else:
        replay_buffer = None
    algo = variant['algo_class'](
        env,
        qf,
        policy,
        exploration_policy,
        replay_buffer=replay_buffer,
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
algo_class_to_sparse_reward = {
    VectorizedTauSdql: False,
    VectorizedDeltaTauSdql: False,
    HorizonFedStateDistanceQLearning: True,
    StateDistanceQLearning: True,
}
algo_class_to_discount = {
    VectorizedTauSdql: 10,
    VectorizedDeltaTauSdql: 10,
    HorizonFedStateDistanceQLearning: 10,
    StateDistanceQLearning: 0.99,
}

experiments=[
    'joint_angle|fixed_angle',
    'joint_angle|varying_angle',
    'end_effector_position|fixed_ee',
    'end_effector_position|varying_ee',
    'end_effector_position_orientation|fixed_ee',
    'end_effector_position_orientation|varying_ee'
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    use_gpu = True
    max_path_length = 100
    # algo_class = VectorizedDeltaTauSdql  # <-- should work well enough
    algo_class = VectorizedTauSdql # <-- Try this if Delta version does not work
    replay_buffer_size = 200000
    reaching_task_variant = dict(
        algo_params=dict(
            num_epochs=60,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=1,
            use_soft_update=True,
            tau=0.001,
            batch_size=64,
            discount=algo_class_to_discount[algo_class],
            sample_train_goals_from='her',
            sample_rollout_goals_from='replay_buffer',
            sample_discount=True,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            replay_buffer_size=replay_buffer_size,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            render=args.render,
            save_replay_buffer=True,
            sparse_reward=algo_class_to_sparse_reward[algo_class],
            cycle_taus_for_rollout=True,
            collect_data=False,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.softplus,
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        normalize_params=dict(),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        algo_class=algo_class,
        qf_class=algo_class_to_qf_class[algo_class],
        her_replay_buffer_params=dict(
            max_size=replay_buffer_size,
            num_goals_to_sample=4,
            goal_sample_strategy='store',
        ),
        env_params={
                      'arm_name': 'right',
                      'safety_box': True,
                      'loss': 'huber',
                      'huber_delta': 10,
                      'safety_force_magnitude': 7,
                      'temp': 15,
                      'remove_action': False,
                      'experiment': experiments[2],
                      'reward_magnitude': 10,
                      'use_safety_checks': False,
            },
    )
    lego_block_task_variant = dict(
        algo_params=dict(
            num_epochs=15,
            num_steps_per_epoch=1000,
            num_steps_per_eval=1000,
            num_updates_per_env_step=1,
            use_soft_update=True,
            tau=0.001,
            batch_size=64,
            discount=algo_class_to_discount[algo_class],
            sample_train_goals_from='her',
            sample_rollout_goals_from='environment',
            sample_discount=True,
            qf_weight_decay=0.,
            max_path_length=max_path_length,
            replay_buffer_size=replay_buffer_size,
            prob_goal_state_is_next_state=0,
            termination_threshold=0,
            render=args.render,
            save_replay_buffer=True,
            sparse_reward=algo_class_to_sparse_reward[algo_class],
            cycle_taus_for_rollout=True,
            collect_data=True,
        ),
        qf_params=dict(
            hidden_sizes=[300, 300],
            hidden_activation=F.softplus,
        ),
        policy_params=dict(
            fc1_size=300,
            fc2_size=300,
        ),
        normalize_params=dict(),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
        algo_class=algo_class,
        qf_class=algo_class_to_qf_class[algo_class],
        her_replay_buffer_params=dict(
            max_size=replay_buffer_size,
            num_goals_to_sample=4,
            goal_sample_strategy='store',
        ),
        env_params={
                      'arm_name': 'right',
                      'safety_box': True,
                      'loss': 'huber',
                      'huber_delta': 10,
                      'safety_force_magnitude': 6,
                      'temp': 15,
                      'remove_action': False,
                      'experiment': experiments[2],
                      'reward_magnitude': 10,
                      'use_safety_checks': False,
            },
    )
    for i in range(5):
        algo_class = reaching_task_variant['algo_class']
        run_experiment(
            experiment,
            seed=random.randint(0, 666),
            exp_prefix="sdql-sawyer-final",
            mode="local",
            variant=reaching_task_variant,
            exp_id=0,
            use_gpu=use_gpu,
            snapshot_mode="last",
        )
