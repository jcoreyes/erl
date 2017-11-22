import argparse
import random
from pathlib import Path

import joblib
from railrl.tf.state_distance.vectorized_sdql import VectorizedDeltaTauSdql, \
    VectorizedTauSdql

import railrl.torch.pytorch_util as ptu
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.state_distance.exploration import \
    UniversalPolicyWrappedWithExplorationStrategy
from railrl.state_distance.networks import (
    FlatUniversalQfunction,
    GoalConditionedDeltaModel, VectorizedGoalStructuredUniversalQfunction,
    GoalStructuredUniversalQfunction)
from railrl.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
    HorizonFedStateDistanceQLearning)
from railrl.torch.modules import HuberLoss


def experiment(variant):
    path = variant['path']
    data = joblib.load(path)
    env = data['env']
    qf = data['qf']
    policy = data['policy']
    action_space = convert_gym_space(env.action_space)
    es = variant['es_class'](
        action_space=action_space,
        **variant['es_params']
    )
    exploration_policy = UniversalPolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    newpath = Path(path).parent / 'extra_data.pkl'
    extra_data = joblib.load(str(newpath))
    replay_buffer = extra_data.get('replay_buffer', None)
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
    algo.train_offline()


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
    parser.add_argument('path', type=str,
                        help='Path to snapshot file to fine tune.')
    args = parser.parse_args()

    n_seeds = 1
    use_gpu = True
    max_path_length = 100
    # algo_class = VectorizedDeltaTauSdql  # <-- should work well enough
    algo_class = VectorizedTauSdql # <-- Try this if Delta version does not work
    replay_buffer_size = 200000
    variant = dict(
        path=args.path,
        algo_params=dict(
            num_epochs=100,
            num_updates_per_env_step=10000,
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
        ),
        algo_class=algo_class,
        es_class=OUStrategy,
        es_params=dict(
            theta=0.1,
            max_sigma=0.25,
            min_sigma=0.25,
        ),
    )
    algo_class = variant['algo_class']
    run_experiment(
        experiment,
        seed=random.randint(0, 666),
        exp_prefix="offline-sdql-sawyer-10K-updatesperepoch",
        mode="local",
        variant=variant,
        exp_id=0,
        use_gpu=use_gpu,
        snapshot_mode="last",
    )
