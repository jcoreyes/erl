"""
Use an oracle qfunction to train a policy in bptt-ddpg style.
"""
import random

import numpy as np
import tensorflow as tf
from hyperopt import hp

from rlkit.data_management.ocm_subtraj_replay_buffer import (
    OcmSubtrajReplayBuffer
)
from rlkit.envs.memory.high_low import HighLow
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.launchers.algo_launchers import bptt_ddpg_launcher
from rlkit.launchers.launcher_util import (
    run_experiment,
)
from rlkit.launchers.launcher_util import (
    run_experiment_here,
    create_base_log_dir,
)
from rlkit.misc.hyperparameter import (
    DeterministicHyperparameterSweeper,
    RandomHyperparameterSweeper,
    LogFloatParam,
    LinearFloatParam,
)
from rlkit.misc.hypopt import optimize_and_save
from rlkit.tf.bptt_ddpg import BpttDDPG
from rlkit.tf.ddpg import TargetUpdateMode
from rlkit.tf.policies.memory.lstm_memory_policy import (
    SeparateLstmLinearCell,
)


def get_ocm_score(variant):
    algorithm = bptt_ddpg_launcher(variant)
    scores = algorithm.epoch_scores
    return np.mean(scores[-3:])


def create_run_experiment_multiple_seeds(n_seeds):
    def run_experiment_with_multiple_seeds(variant):
        scores = []
        for i in range(n_seeds):
            variant['seed'] = str(int(variant['seed']) + i)
            exp_prefix = variant['exp_prefix']
            scores.append(run_experiment_here(
                get_ocm_score,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=i,
            ))
        return np.mean(scores)

    return run_experiment_with_multiple_seeds

if __name__ == '__main__':
    n_seeds = 1
    mode = 'here'
    # exp_prefix = "dev-bptt-ddpg"
    exp_prefix = "6-8-dev-bptt-ddpg-tf"
    run_mode = 'none'
    version = 'dev'
    num_hp_settings = 100

    # n_seeds = 8
    # mode = 'ec2'
    # exp_prefix = "6-6-hl-bptt-ddpg-rwa-grid-memorydim-nbptt-target-mode"
    # version = 'Our Method - Half BPTT (dev)'

    # run_mode = 'grid'
    """
    Miscellaneous Params
    """
    oracle_mode = 'none'
    algo_class = BpttDDPG
    load_policy_file = (
        '/home/vitchyr/git/rllab-rail/railrl/data/reference/expert'
        '/ocm_reward_magnitude5_H6_nbptt6_100p'
        '/params.pkl'
    )
    load_policy_file = None

    """
    Set all the hyperparameters!
    """
    env_class = HighLow
    # env_class = WaterMazeEasy
    H = 16
    num_steps_per_iteration = 100
    num_iterations = 50

    eval_samples = 400
    env_params = dict(
        num_steps=H,
        position_only=True,
    )

    # TODO(vitchyr): clean up this hacky dropout code. Also, you'll need to
    # fix the batchnorm code. Basically, calls to (e.g.) qf.output will
    # always take the eval output.

    # noinspection PyTypeChecker
    ddpg_params = dict(
        batch_size=32,
        n_epochs=num_iterations,
        n_updates_per_time_step=1,
        epoch_length=num_steps_per_iteration,
        eval_samples=eval_samples,
        max_path_length=H,
        discount=1.0,
        save_tf_graph=False,
        num_steps_between_train=1,
        # Target network
        soft_target_tau=0.01,
        hard_update_period=100,
        # target_update_mode=TargetUpdateMode.HARD,
        target_update_mode=TargetUpdateMode.SOFT,
        # QF hyperparameters
        qf_learning_rate=1e-3,
        num_extra_qf_updates=0,
        extra_qf_training_mode='fixed',
        extra_train_period=100,
        qf_weight_decay=0,
        qf_total_loss_tolerance=0.03,
        train_qf_on_all=True,
        dropout_keep_prob=1.,
        # Policy hps
        policy_learning_rate=1e-3,
        max_num_q_updates=1000,
        train_policy=True,
        write_policy_learning_rate=1e-5,
        train_policy_on_all_qf_timesteps=True,
        # write_only_optimize_bellman=True,
        # env_action_minimize_bellman_loss=True,
        write_only_optimize_bellman=False,
        env_action_minimize_bellman_loss=False,
        # memory
        num_bptt_unrolls=16,
        # bpt_bellman_error_weight=10,
        bpt_bellman_error_weight=0,
        reward_low_bellman_error_weight=0.,
        saved_write_loss_weight=0,
        # Replay buffer
        replay_pool_size=100000,
        compute_gradients_immediately=False,
        # TODO: figure out why this matters if we're doing full BPTT
        save_new_memories_back_to_replay_buffer=True,
        refresh_entire_buffer_period=None,
    )

    # noinspection PyTypeChecker
    policy_params = dict(
        # rnn_cell_class=LstmLinearCell,
        rnn_cell_class=SeparateLstmLinearCell,
        # rnn_cell_class=LstmLinearCellNoiseAll,
        # rnn_cell_class=DebugCell,
        # rnn_cell_class=SeparateRWALinearCell,
        rnn_cell_params=dict(
            use_peepholes=True,
            env_noise_std=0,
            memory_noise_std=0,
            output_nonlinearity=tf.nn.tanh,
            env_hidden_sizes=[100, 100],
            env_hidden_activation=tf.nn.tanh,
        )
    )

    oracle_params = dict(
        env_grad_distance_weight=0.,
        write_grad_distance_weight=0.,
        qf_grad_mse_from_one_weight=0.,
        regress_onto_values_weight=0.,
        bellman_error_weight=1.,
        use_oracle_qf=False,
        unroll_through_target_policy=False,
    )

    meta_qf_params = dict(
        use_time=False,
        use_target=False,
    )
    meta_params = dict(
        meta_qf_learning_rate=0.0001900271829580542,
        meta_qf_output_weight=0,
        qf_output_weight=1,
    )

    # noinspection PyTypeChecker
    es_params = dict(
        # env_es_class=NoopStrategy,
        env_es_class=OUStrategy,
        env_es_params=dict(
            max_sigma=1,
            min_sigma=None,
        ),
        # memory_es_class=NoopStrategy,
        memory_es_class=OUStrategy,
        memory_es_params=dict(
            max_sigma=1,
            min_sigma=None,
        ),
        noise_action_to_memory=False,
    )

    # noinspection PyTypeChecker
    qf_params = dict(
        # hidden_nonlinearity=tf.nn.relu,
        # output_nonlinearity=tf.nn.tanh,
        # hidden_nonlinearity=tf.identity,
        # output_nonlinearity=tf.identity,
        # embedded_hidden_sizes=[100, 64, 32],
        # observation_hidden_sizes=[100],
        use_time=False,
        use_target=False,
        use_dropout=True,
    )


    """
    Create monolithic variant dictionary
    """
    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        memory_dim=20,
        exp_prefix=exp_prefix,
        algo_class=algo_class,
        version=version,
        load_policy_file=load_policy_file,
        oracle_mode=oracle_mode,
        env_class=env_class,
        env_params=env_params,
        ddpg_params=ddpg_params,
        policy_params=policy_params,
        qf_params=qf_params,
        meta_qf_params=meta_qf_params,
        oracle_params=oracle_params,
        es_params=es_params,
        meta_params=meta_params,
        replay_buffer_class=OcmSubtrajReplayBuffer,
        # replay_buffer_class=UpdatableSubtrajReplayBuffer,
        replay_buffer_params=dict(
            keep_old_fraction=0.9,
        ),
        memory_aug_params=dict(
            max_magnitude=1e6,
        ),
    )

    if run_mode == 'hyperopt':
        search_space = {
            'policy_params.rnn_cell_params.env_noise_std': hp.uniform(
                'policy_params.rnn_cell_params.env_noise_std',
                0.,
                5,
            ),
            'policy_params.rnn_cell_params.memory_noise_std': hp.uniform(
                'policy_params.rnn_cell_params.memory_noise_std',
                0.,
                5,
            ),
            'ddpg_params.bpt_bellman_error_weight': hp.loguniform(
                'ddpg_params.bpt_bellman_error_weight',
                np.log(0.01),
                np.log(1000),
            ),
            'ddpg_params.qf_learning_rate': hp.loguniform(
                'ddpg_params.qf_learning_rate',
                np.log(0.00001),
                np.log(0.01),
            ),
            'meta_params.meta_qf_learning_rate': hp.loguniform(
                'meta_params.meta_qf_learning_rate',
                np.log(1e-5),
                np.log(1e-2),
            ),
            'meta_params.meta_qf_output_weight': hp.loguniform(
                'meta_params.meta_qf_output_weight',
                np.log(1e-1),
                np.log(1000),
            ),
            'seed': hp.randint('seed', 10000),
        }

        base_log_dir = create_base_log_dir(exp_prefix=exp_prefix)

        optimize_and_save(
            base_log_dir,
            create_run_experiment_multiple_seeds(n_seeds=n_seeds),
            search_space=search_space,
            extra_function_kwargs=variant,
            maximize=True,
            verbose=True,
            load_trials=True,
            num_rounds=500,
            num_evals_per_round=1,
        )
    elif run_mode == 'grid':
        search_space = {
            'memory_dim': [4, 16, 80],
            # 'policy_params.rnn_cell_class': [
            #     SeparateLstmLinearCell,
            #     SeparateRWALinearCell,
            # ],
            # 'policy_params.rnn_cell_params.env_noise_std': [0.1, 0.3, 1.],
            # 'policy_params.rnn_cell_params.memory_noise_std': [0.1, 0.3, 1.],
            # 'policy_params.rnn_cell_params.env_hidden_sizes': [
            #     [],
            #     [32],
            #     [32, 32],
            # ],
            # 'qf_params.embedded_hidden_sizes': [
            #     [100, 64, 32],
            #     [100],
            #     [32],
            # ],
            # 'ddpg_params.dropout_keep_prob': [1, 0.9, 0.5],
            # 'ddpg_params.qf_weight_decay': [0, 0.001],
            # 'ddpg_params.reward_low_bellman_error_weight': [0, 0.1, 1., 10.],
            # 'ddpg_params.num_extra_qf_updates': [0, 5],
            # 'ddpg_params.batch_size': [512, 128, 32, 8],
            # 'ddpg_params.replay_pool_size': [900, 90000],
            'ddpg_params.num_bptt_unrolls': [32, 16],
            # 'ddpg_params.n_updates_per_time_step': [1, 10],
            # 'ddpg_params.policy_learning_rate': [1e-3, 1e-4],
            # 'ddpg_params.write_policy_learning_rate': [1e-4, 1e-5],
            # 'ddpg_params.hard_update_period': [1, 100, 1000],
            # 'ddpg_params.soft_target_tau': [0.001, 0.01, 1],
            # 'ddpg_params.bpt_bellman_error_weight': [1, 10],
            # 'ddpg_params.saved_write_loss_weight': [1, 10],
            # 'ddpg_params.env_action_minimize_bellman_loss': [False, True],
            # 'ddpg_params.save_new_memories_back_to_replay_buffer': [True,
            #                                                         False],
            # 'ddpg_params.refresh_entire_buffer_period': [1, None],
            # 'ddpg_params.write_only_optimize_bellman': [False, True],
            # 'ddpg_params.discount': [1.0, 0.9],
            'ddpg_params.target_update_mode': [
                TargetUpdateMode.SOFT,
                TargetUpdateMode.HARD,
            ],
            # 'meta_params.meta_qf_learning_rate': [1e-3, 1e-4],
            # 'meta_params.meta_qf_output_weight': [0.1, 1, 10],
            # 'meta_params.qf_output_weight': [0, 1],
            # 'env_params.episode_boundary_flags': [True, False],
            # 'env_params.num_steps': [12, 16, 24],
            # 'es_params.memory_es_class': [GaussianStrategy, OUStrategy],
            # 'es_params.env_es_class': [GaussianStrategy, OUStrategy],
            # 'es_params.memory_es_params.max_sigma': [0.1, 0.3, 1],
            # 'es_params.memory_es_params.min_sigma': [1],
            # 'es_params.env_es_params.max_sigma': [0.1, 0.3, 1],
            # 'es_params.env_es_params.min_sigma': [1],
            # 'replay_buffer_params.keep_old_fraction': [0, 0.5, 0.9],
        }
        sweeper = DeterministicHyperparameterSweeper(search_space,
                                                     default_parameters=variant)
        for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
            for i in range(n_seeds):
                run_experiment(
                    get_ocm_score,
                    exp_prefix=exp_prefix,
                    seed=i,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    elif run_mode == 'random':
        sweeper = RandomHyperparameterSweeper(
            hyperparameters=[
                LinearFloatParam(
                    'policy_params.rnn_cell_params.env_noise_std', 0, 1
                ),
                LinearFloatParam(
                    'policy_params.rnn_cell_params.memory_noise_std', 0, 1
                ),
                LogFloatParam(
                    'ddpg_params.bpt_bellman_error_weight', 1, 1001, offset=-1
                ),
                LogFloatParam('meta_params.meta_qf_learning_rate', 1e-5, 1e-2),
                LogFloatParam(
                    'meta_params.meta_qf_output_weight', 1e-3, 1e3, offset=-1e-3
                ),
            ],
            default_kwargs=variant,
        )
        for exp_id in range(num_hp_settings):
            variant = sweeper.generate_random_hyperparameters()
            for i in range(n_seeds):
                run_experiment(
                    get_ocm_score,
                    exp_prefix=exp_prefix,
                    seed=i,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    elif run_mode == 'custom_grid':
        for exp_id, (
                version,
                subseq_length,
        ) in enumerate([
            ("num_steps_per_batch=256", 32),
            ("num_steps_per_batch=256", 16),
            ("num_steps_per_batch=256", 8),
            ("num_steps_per_batch=256", 4),
            ("num_steps_per_batch=256", 1),
        ]):
            num_steps_per_batch = 256
            batch_size = int(num_steps_per_batch / subseq_length)
            variant['version'] = version
            variant['ddpg_params']['num_bptt_unrolls'] = subseq_length
            variant['ddpg_params']['batch_size'] = batch_size
            variant['ddpg_params']['min_pool_size'] = batch_size
            for seed in range(n_seeds):
                run_experiment(
                    get_ocm_score,
                    exp_prefix=exp_prefix,
                    seed=seed,
                    mode=mode,
                    variant=variant,
                    exp_id=exp_id,
                )
    elif run_mode == 'ablation':
        pass
    else:
        for _ in range(n_seeds):
            seed = random.randint(0, 10000)
            run_experiment(
                bptt_ddpg_launcher,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=0,
            )
