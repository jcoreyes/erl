"""
Our method.
"""
import random

import tensorflow as tf

from railrl.algos.bptt_ddpg import BpttDDPG
from railrl.algos.ddpg import TargetUpdateMode
from railrl.data_management.ocm_subtraj_replay_buffer import (
    OcmSubtrajReplayBuffer
)
from railrl.envs.memory.high_low import HighLow
from railrl.envs.water_maze import WaterMaze, WaterMazeEasy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.launchers.algo_launchers import bptt_ddpg_launcher
from railrl.launchers.launcher_util import (
    run_experiment,
)
from railrl.policies.memory.lstm_memory_policy import (
    SeparateLstmLinearCell,
)

if __name__ == '__main__':
    n_seeds = 1
    mode = 'here'
    exp_prefix = "dev-bptt-ddpg-ocm"

    # n_seeds = 10
    # mode = 'ec2'
    # exp_prefix = '5-27-benchmark-sl-highlow'

    """
    Miscellaneous Params
    """
    oracle_mode = 'none'
    algo_class = BpttDDPG
    # algo_class = NoOpBpttDDPG
    load_policy_file = (
        '/home/vitchyr/git/rllab-rail/railrl/data/reference/expert'
        '/ocm_reward_magnitude5_H6_nbptt6_100p'
        '/params.pkl'
    )
    load_policy_file = None

    """
    Set all the hyperparameters!
    """
    # env_class = WaterMazeEasy
    # env_class = WaterMaze
    env_class = HighLow
    if env_class == WaterMaze:
        env_params = dict(
            num_steps=200,
        )
        epoch_length = 10000
        eval_samples = 2000
    elif env_class == HighLow:
        env_params = dict(
            num_steps=32,
        )
        epoch_length = 1000
        eval_samples = 200
    else:
        raise Exception("Invalid env_class: %s" % env_class)


    # noinspection PyTypeChecker
    ddpg_params = dict(
        batch_size=32,
        n_epochs=100,
        min_pool_size=32,
        replay_pool_size=100000,
        n_updates_per_time_step=5,
        epoch_length=epoch_length,
        eval_samples=eval_samples,
        max_path_length=1002,
        discount=1.0,
        save_tf_graph=False,
        # Target network
        soft_target_tau=0.01,
        hard_update_period=1000,
        target_update_mode=TargetUpdateMode.HARD,
        # QF hyperparameters
        qf_learning_rate=1e-3,
        num_extra_qf_updates=0,
        extra_qf_training_mode='fixed',
        extra_train_period=100,
        qf_weight_decay=0,
        qf_total_loss_tolerance=0.03,
        train_qf_on_all=False,
        dropout_keep_prob=1.,
        # Policy hps
        policy_learning_rate=1e-3,
        max_num_q_updates=1000,
        train_policy=True,
        write_policy_learning_rate=1e-5,
        train_policy_on_all_qf_timesteps=False,
        write_only_optimize_bellman=True,
        # memory
        num_bptt_unrolls=32,
        bpt_bellman_error_weight=10,
        reward_low_bellman_error_weight=0.,
        saved_write_loss_weight=0,
        compute_gradients_immediately=False,
    )

    # noinspection PyTypeChecker
    policy_params = dict(
        rnn_cell_class=SeparateLstmLinearCell,
        rnn_cell_params=dict(
            use_peepholes=True,
            env_noise_std=.0,
            memory_noise_std=0.,
            output_nonlinearity=tf.nn.tanh,
            env_hidden_sizes=[],
            # env_hidden_activation=tf.tanh,
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

    memory_dim = 20
    replay_buffer_params = dict(
        keep_old_fraction=0.9,
    )

    """
    Create monolithic variant dictionary
    """
    # noinspection PyTypeChecker
    variant = dict(
        memory_dim=memory_dim,
        exp_prefix=exp_prefix,
        algo_class=algo_class,
        version="Our Method - Full BPTT",
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
        replay_buffer_params=replay_buffer_params,
    )
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
