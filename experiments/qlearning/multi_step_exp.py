"""
Multi-step Q-learning.
"""
import random
import numpy as np

from rlkit.envs.env_utils import gym_env
from rlkit.envs.time_limited_env import TimeLimitedEnv
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.launcher_util import run_experiment
import rlkit.misc.hyperparameter as hyp
from rlkit.torch.networks import FeedForwardQFunction, FeedForwardPolicy
from rlkit.torch.ddpg.multi_step_ql import MultiStepDdpg
from rllab.envs.normalized_env import normalize

from hyperopt import hp
from rlkit.misc.hypopt import optimize_and_save
from rlkit.launchers.launcher_util import (
    create_base_log_dir,
    create_run_experiment_multiple_seeds,
)


def experiment(variant):
    # env = HalfCheetahEnv()
    # env = PointEnv()
    env = gym_env("Pendulum-v0")
    # env = HopperEnv()
    horizon = variant['algo_params']['max_path_length']
    env = TimeLimitedEnv(env, horizon)
    env = normalize(env)
    es = OUStrategy(action_space=env.action_space)
    qf = FeedForwardQFunction(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    policy = FeedForwardPolicy(
        int(env.observation_space.flat_dim),
        int(env.action_space.flat_dim),
        400,
        300,
    )
    algorithm = MultiStepDdpg(
        env,
        exploration_strategy=es,
        qf=qf,
        policy=policy,
        **variant['algo_params']
    )
    algorithm.train()
    return algorithm.final_score


if __name__ == "__main__":
    num_configurations = 1  # for random mode

    n_seeds = 1
    mode = "here"
    exp_prefix = "7-7-dev-multi-step-ql"
    version = "dev"
    run_mode = "none"

    # n_seeds = 100
    # mode = "ec2"
    # exp_prefix = "6-28-multi-step-ql"
    # version = "Multistep QL"

    # run_mode = 'random'
    use_gpu = True
    if mode != "here":
        use_gpu = False

    # noinspection PyTypeChecker
    variant = dict(
        algo_params=dict(
            num_epochs=100,
            num_steps_per_epoch=10000,
            num_steps_per_eval=1000,
            batch_size=128,
            max_path_length=1000,
            target_hard_update_period=100,
            discount=0.98,
            policy_learning_rate=4e-4,
            qf_learning_rate=2e-3,
            subtraj_length=10,
        ),
        version=version,
    )
    if run_mode == 'grid':
        search_space = {
            'algo_params.discount': [0.99, 0.9, 0.5],
            'algo_params.policy_learning_rate': [1e-4, 1e-3, 1e-2],
            'algo_params.qf_learning_rate': [1e-4, 1e-3, 1e-2],
            'algo_params.target_hard_update_period': [10, 100, 1000],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(
            search_space, default_parameters=variant
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
                    periodic_sync_interval=600,
                )
    if run_mode == 'random':
        hyperparameters = [
            hyp.LinearFloatParam('algo_params.discount', 0, 1),
            hyp.LogFloatParam('algo_params.policy_learning_rate', 1e-7, 1e-1),
            hyp.LogFloatParam('algo_params.qf_learning_rate', 1e-7, 1e-1),
            hyp.LogIntParam('algo_params.target_hard_update_period', 1, 1000),
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
                    periodic_sync_interval=600,
                )
    if run_mode == 'hyperopt':
        search_space = {
            'algo_params.qf_learning_rate': hp.loguniform(
                'algo_params.qf_learning_rate',
                np.log(1e-5),
                np.log(1e-2),
            ),
            'algo_params.policy_learning_rate': hp.loguniform(
                'algo_params.policy_learning_rate',
                np.log(1e-5),
                np.log(1e-2),
            ),
            'algo_params.discount': hp.uniform(
                'algo_params.discount',
                0.0,
                1.0,
            ),
            'algo_params.target_hard_update_period': hp.qloguniform(
                'algo_params.target_hard_update_period',
                np.log(1),
                np.log(1000),
                1,
            ),
            'seed': hp.randint('seed', 10000),
        }

        base_log_dir = create_base_log_dir(exp_prefix=exp_prefix)
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
                periodic_sync_interval=600,
            )
