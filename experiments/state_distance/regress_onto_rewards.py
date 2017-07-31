import argparse
import pickle
import random

import numpy as np
from hyperopt import hp

import railrl.misc.hyperparameter as hyp
import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    StateDistanceQLearning,
)
from railrl.algos.state_distance.supervised_learning import SupervisedLearning
from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.split_buffer import SplitReplayBuffer
from railrl.envs.multitask.reacher_env import (
    GoalStateSimpleStateReacherEnv,
    XyMultitaskSimpleStateReacherEnv)
from railrl.envs.wrappers import convert_gym_space
from railrl.exploration_strategies.gaussian_strategy import GaussianStrategy
from railrl.launchers.launcher_util import (
    create_log_dir,
    create_run_experiment_multiple_seeds,
)
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.hypopt import optimize_and_save
from railrl.misc.ml_util import RampUpSchedule
from railrl.networks.state_distance import UniversalQfunction
from railrl.policies.zero_policy import ZeroPolicy
from railrl.samplers.path_sampler import MultitaskPathSampler


def experiment(variant):
    env_class = variant['env_class']
    env = env_class(**variant['env_params'])
    if variant['generate_data']:
        action_space = convert_gym_space(env.action_space)
        es = GaussianStrategy(
            action_space=action_space,
            max_sigma=0.2,
            min_sigma=0.2,
        )
        exploration_policy = ZeroPolicy(
            int(action_space.flat_dim),
        )
        sampler_params = variant['sampler_params']
        replay_buffer_size = (
            sampler_params['min_num_steps_to_collect']
            + sampler_params['max_path_length']
        )
        replay_buffer = SplitReplayBuffer(
            EnvReplayBuffer(
                replay_buffer_size,
                env,
                flatten=True,
            ),
            EnvReplayBuffer(
                replay_buffer_size,
                env,
                flatten=True,
            ),
            fraction_paths_in_train=0.8,
        )
        sampler = MultitaskPathSampler(
            env,
            exploration_strategy=es,
            exploration_policy=exploration_policy,
            replay_buffer=replay_buffer,
            **variant['sampler_params']
        )
        sampler.collect_data()
        replay_buffer = sampler.replay_buffer
    else:
        dataset_path = variant['dataset_path']
        with open(dataset_path, 'rb') as handle:
            replay_buffer = pickle.load(handle)

    observation_space = convert_gym_space(env.observation_space)
    action_space = convert_gym_space(env.action_space)
    qf = UniversalQfunction(
        int(observation_space.flat_dim),
        int(action_space.flat_dim),
        env.goal_dim,
        **variant['qf_params']
    )
    algo = SupervisedLearning(
        env,
        qf,
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
    args = parser.parse_args()

    num_configurations = 1  # for random mode
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-sdqlr-supervised-learning"
    version = "Dev"
    run_mode = "none"
    snapshot_mode = "last"
    snapshot_gap = 5

    # n_seeds = 3
    # mode = "ec2"
    # exp_prefix = "sdqlr-sweep-lr-2"

    # run_mode = 'grid'
    use_gpu = True
    if mode != "here":
        use_gpu = False

    dataset_path = args.replay_path

    # noinspection PyTypeChecker
    variant = dict(
        dataset_path=str(dataset_path),
        algo_params=dict(
            num_epochs=101,
            num_batches_per_epoch=1000,
            num_unique_batches=1000,
            batch_size=100,
            qf_learning_rate=1e-4,
        ),
        qf_params=dict(
            hidden_sizes=[400, 300],
        ),
        epoch_discount_schedule_class=RampUpSchedule,
        epoch_discount_schedule_params=dict(
            min_value=0.,
            max_value=0.,
            ramp_duration=100,
        ),
        env_class=GoalStateSimpleStateReacherEnv,
        # env_class=XyMultitaskSimpleStateReacherEnv,
        env_params=dict(
            add_noop_action=False,
            reward_weights=[1, 1, 1, 1, 0, 0],
        ),
        sampler_params=dict(
            min_num_steps_to_collect=20000,
            max_path_length=1000,
            render=False,
        ),
        generate_data=args.replay_path is None,
    )
    if run_mode == 'grid':
        search_space = {
            'algo_params.qf_learning_rate': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            'qf_params.hidden_sizes': [
                [100, 100],
                [400, 300],
                [400, 300, 200],
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
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    periodic_sync_interval=3600,
                    snapshot_mode=snapshot_mode,
                    snapshot_gap=snapshot_gap,
                )
    if run_mode == 'hyperopt':
        search_space = {
            'float_param': hp.uniform(
                'float_param',
                0.,
                5,
            ),
            'float_param2': hp.loguniform(
                'float_param2',
                np.log(0.01),
                np.log(1000),
            ),
            'seed': hp.randint('seed', 10000),
        }

        base_log_dir = create_log_dir(exp_prefix=exp_prefix)

        optimize_and_save(
            base_log_dir,
            create_run_experiment_multiple_seeds(
                n_seeds,
                experiment,
                exp_prefix=exp_prefix,
            ),
            search_space=search_space,
            extra_function_kwargs=variant,
            maximize=True,
            verbose=True,
            load_trials=True,
            num_rounds=500,
            num_evals_per_round=1,
        )
    if run_mode == 'random':
        hyperparameters = [
            hyp.LinearFloatParam('foo', 0, 1),
            hyp.LogFloatParam('bar', 1e-5, 1e2),
        ]
        sweeper = hyp.RandomHyperparameterSweeper(
            hyperparameters,
            default_kwargs=variant,
        )
        for _ in range(num_configurations):
            for exp_id in range(n_seeds):
                seed = random.randint(0, 10000)
                variant = sweeper.generate_random_hyperparameters()
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                    sync_s3_log=True,
                    sync_s3_pkl=True,
                    periodic_sync_interval=3600,
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
                periodic_sync_interval=3600,
                snapshot_mode=snapshot_mode,
                snapshot_gap=snapshot_gap,
            )
