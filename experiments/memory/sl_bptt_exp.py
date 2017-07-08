"""
Supervised learning BPTT on OCM.
"""
from itertools import product
import tensorflow as tf
import random

from railrl.core.rnn.rnn import RWACell
from railrl.envs.memory.high_low import HighLow
from railrl.launchers.rnn_launchers import bptt_launcher
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from tensorflow.contrib.rnn import (
    LayerNormBasicLSTMCell,
    LSTMCell,
    BasicLSTMCell,
    GRUCell,
)
from railrl.policies.memory.action_aware_memory_policy import DecoupledLSTM
from railrl.policies.memory.lstm_memory_policy import (
    LstmLinearCell,
    LstmLinearCellNoiseAll,
    LstmLinearCellSwapped,
    LstmLinearCellNoiseAllNoiseLogit,
    ResidualLstmLinearCell,
    GRULinearCell,
    SeparateLstmLinearCell,
    FfResCell,
    SeparateRWALinearCell,
)


def main():
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-sl"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "6-2-sl-rwa-vs-lstm"

    num_seeds = 5
    num_values = 2
    use_peepholes = True
    rnn_cell_class = LstmLinearCell
    rnn_cell_class = ResidualLstmLinearCell
    softmax = False
    version = 'supervised_learning'
    exp_prefix = '5-11-sl-noise-architecture-sweep'
    # exp_prefix = 'dev-sl'
    env_noise_std = 0.
    memory_noise_std = .1
    mode = 'here'
    for H, rnn_cell_class, env_noise_std, memory_noise_std in product(
        [32],
        [FfResCell],
        [0],
        [0],
    ):
        # noinspection PyTypeChecker
        variant = dict(
            H=H,
            exp_prefix=exp_prefix,
            algo_params=dict(
                num_batches_per_epoch=10000//32,
                num_epochs=100,
                learning_rate=1e-3,
                batch_size=32,
                eval_num_episodes=64,
                lstm_state_size=10,
                rnn_cell_class=rnn_cell_class,
                rnn_cell_params=dict(
                    use_peepholes=use_peepholes,
                    env_noise_std=env_noise_std,
                    memory_noise_std=memory_noise_std,
                    # output_nonlinearity=tf.nn.softmax,
                    env_output_nonlinearity=tf.nn.softmax,
                    # env_hidden_sizes=[100],
                    # env_hidden_activation=tf.identity,
                    # write_output_nonlinearity=tf.identity,
                    # write_hidden_sizes=[100],
                    # write_hidden_activation=tf.identity,
                ),
                # rnn_cell_class=SeparateLstmLinearCell,
                # rnn_cell_params=dict(
                #     use_peepholes=True,
                #     env_noise_std=env_noise_std,
                #     memory_noise_std=memory_noise_std,
                #     output_nonlinearity=tf.nn.tanh,
                #     # output_nonlinearity=tf.nn.softmax,
                #     env_hidden_sizes=[],
                # ),
                softmax=False,
            ),
            version='Supervised Learning',
            env_class=HighLow,
            # env_class=OneCharMemory,
        )

        exp_id = -1
        for seed in range(n_seeds):
            exp_id += 1
            set_seed(seed)
            variant['seed'] = seed
            variant['exp_id'] = exp_id

            run_experiment(
                bptt_launcher,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
            )


if __name__ == "__main__":
    main()
