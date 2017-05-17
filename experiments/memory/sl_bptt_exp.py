"""
Supervised learning BPTT on OCM.
"""
from itertools import product
import random
from railrl.launchers.rnn_launchers import bptt_launcher
from railrl.launchers.launcher_util import run_experiment
import tensorflow as tf
from tensorflow.contrib.rnn import (
    LayerNormBasicLSTMCell,
    LSTMCell,
    BasicLSTMCell,
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
)




def main():
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
        variant = dict(
            env_params=dict(
                env_id='ocm',
                normalize_env=False,
                init_env_params=dict(
                    num_steps=H,
                    n=num_values,
                )
            ),
            algo_params=dict(
                num_batches_per_epoch=100,
                num_epochs=20,
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
                softmax=softmax,
            ),
            version=version,
        )
        for _ in range(num_seeds):
            seed = random.randint(0, 100000)
            run_experiment(
                bptt_launcher,
                # exp_prefix="dev-ocm-sl",
                # exp_prefix="4-25-ocm-sl-lstm-type-and-peephole--sweep",
                exp_prefix=exp_prefix,
                seed=seed,
                variant=variant,
                mode=mode,
            )


if __name__ == "__main__":
    main()
