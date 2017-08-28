import argparse
import random

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.model_learning import ModelLearning
from railrl.algos.state_distance.util import get_replay_buffer
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
    XyMultitaskSimpleStateReacherEnv,
)
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.predictors.torch import Mlp


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    replay_buffer = get_replay_buffer(variant)

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    model = Mlp(
        int(observation_space.flat_dim) + int(action_space.flat_dim),
        int(observation_space.flat_dim),
        **variant['model_params']
    )
    algo = ModelLearning(
        env,
        model,
        replay_buffer=replay_buffer,
        **variant['algo_params']
    )
    if ptu.gpu_enabled():
        algo.cuda()
    algo.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-reacher-model-learning"
    version = "Dev"
    run_mode = "none"

    # n_seeds = 3
    # mode = "ec2"
    exp_prefix = "reacher-2d-full-goal-learn-model-longer"

    # run_mode = 'grid'
    num_configurations = 1  # for random mode
    snapshot_mode = "gap"
    snapshot_gap = 5
    use_gpu = True
    if mode != "here":
        use_gpu = False

    dataset_path = args.replay_path

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_epochs=100,
            num_batches_per_epoch=1000,
            num_unique_batches=1000,
            batch_size=100,
            learning_rate=1e-3,
        ),
        model_params=dict(
            hidden_sizes=[400, 300],
        ),
        env_class=GoalStateSimpleStateReacherEnv,
        # env_class=PusherEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        env_params=dict(
            # add_noop_action=False,
        ),
        sampler_params=dict(
            min_num_steps_to_collect=10000,
            max_path_length=150,
            render=args.render,
        ),
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            max_sigma=0.2,
            min_sigma=0.2,
        ),
        generate_data=args.replay_path is None,
    )
    seed = random.randint(0, 10000)
    run_experiment(
        experiment,
        exp_prefix=exp_prefix,
        seed=seed,
        mode=mode,
        variant=variant,
        exp_id=0,
        use_gpu=use_gpu,
        sync_s3_log=True,
        sync_s3_pkl=True,
        periodic_sync_interval=3600,
        snapshot_mode=snapshot_mode,
        snapshot_gap=snapshot_gap,
    )
