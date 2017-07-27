import argparse
import pickle
import random
import numpy as np

from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
)
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
)
from railrl.envs.wrappers import convert_gym_space
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.torch import FeedForwardPolicy
from railrl.qfunctions.torch import FeedForwardQFunction


def main(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    dataset_path = variant['dataset_path']
    with open(dataset_path, 'rb') as handle:
        replay_buffer = pickle.load(handle)

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = FeedForwardQFunction(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
        batchnorm_obs=False,
    )
    policy = FeedForwardPolicy(
        int(observation_space.flat_dim) + env.goal_dim,
        int(action_space.flat_dim),
        400,
        300,
    )
    algo = StateDistanceQLearning(
        env=env,
        qf=qf,
        policy=policy,
        replay_buffer=replay_buffer,
        exploration_policy=None,
        **variant['algo_params']
    )
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_path', type=str,
                        help='path to the snapshot file')
    args = parser.parse_args()

    n_seeds = 1
    mode = "here"
    exp_prefix = "7-26-dev-sdql-reacher-full-state-no-bn-add-noop"
    snapshot_mode = 'gap'
    snapshot_gap = 5

    dataset_path = args.replay_path

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_batches=100000,
            num_batches_per_epoch=1000,
            use_soft_update=True,
            tau=1e-3,
            batch_size=1024,
            discount=0.,
            qf_learning_rate=1e-4,
            policy_learning_rate=1e-5,
            sample_goals_from='replay_buffer',
        ),
        env_class=GoalStateSimpleStateReacherEnv,
        env_params=dict(
            add_noop_action=True,
            reward_weights=[1, 1, 1, 1, 0, 0],
        ),
        # env_class=XyMultitaskSimpleStateReacherEnv,
    )

    seed = random.randint(0, 10000)
    run_experiment(
        main,
        exp_prefix=exp_prefix,
        seed=seed,
        mode=mode,
        variant=variant,
        exp_id=0,
        use_gpu=True,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
    )
