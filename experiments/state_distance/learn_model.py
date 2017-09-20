import argparse
import random

import railrl.torch.pytorch_util as ptu
import railrl.misc.hyperparameter as hyp
from railrl.algos.state_distance.model_learning import ModelLearning
from railrl.algos.state_distance.util import get_replay_buffer
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
    XyMultitaskSimpleStateReacherEnv,
)
from railrl.envs.wrappers import convert_gym_space, normalize_box
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.model_based import GreedyModelBasedPolicy
from railrl.predictors.torch import Mlp


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    env = normalize_box(
        env,
        **variant['normalize_params']
    )
    if variant['start_with_empty_replay_buffer']:
        replay_buffer = None
    else:
        replay_buffer = get_replay_buffer(variant)

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    model = Mlp(
        int(observation_space.flat_dim) + int(action_space.flat_dim),
        int(observation_space.flat_dim),
        **variant['model_params']
    )
    policy = GreedyModelBasedPolicy(
        model,
        env,
        sample_size=10000,
    )
    algo = ModelLearning(
        env,
        model,
        replay_buffer=replay_buffer,
        eval_policy=policy,
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

    n_seeds = 3
    mode = "ec2"
    exp_prefix = "reacher-model-learning-grid-search"
    run_mode = 'grid'

    num_configurations = 1  # for random mode
    snapshot_mode = "last"
    snapshot_gap = 5
    use_gpu = True
    if mode != "here":
        use_gpu = False

    dataset_path = args.replay_path

    # noinspection PyTypeChecker
    max_path_length = 300
    replay_buffer_size = 100000
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_epochs=101,
            num_batches_per_epoch=1000,
            num_unique_batches=1000,
            batch_size=100,
            learning_rate=1e-3,
            max_path_length=max_path_length,
            replay_buffer_size=replay_buffer_size,
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
        normalize_params=dict(
            obs_mean=None,
            obs_std=[0.7, 0.7, 0.7, 0.6, 40, 5],
        ),
        sampler_params=dict(
            min_num_steps_to_collect=10000,
            max_path_length=max_path_length,
            render=args.render,
        ),
        replay_buffer_size=replay_buffer_size,
        sampler_es_class=OUStrategy,
        sampler_es_params=dict(
            max_sigma=0.2,
            min_sigma=0.2,
        ),
        generate_data=args.replay_path is None,
        start_with_empty_replay_buffer=True,
    )
    if run_mode == 'grid':
        search_space = {
            'model_params.hidden_sizes': [
                [400, 300],
                [100, 100],
                [32, 32],
            ],
            'algo_params.weight_decay': [0, 1e-4, 1e-3, 1e-2],
            'normalize_params.obs_std': [
                [0.7, 0.7, 0.7, 0.6, 47, 15],
                None,
                [0.7, 0.3, 0.7, 0.3, 25, 5],
            ],
            'env_class': [
                XyMultitaskSimpleStateReacherEnv,
                GoalStateSimpleStateReacherEnv,
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
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    use_gpu=use_gpu,
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    periodic_sync_interval=300,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                )
    else:
        for _ in range(n_seeds):
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
                periodic_sync_interval=300,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
            )
