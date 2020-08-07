"""
Supervised learning with full BPTT.
"""
import random

import tensorflow as tf

from rlkit.envs.memory.high_low import HighLow
from rlkit.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from rlkit.launchers.rnn_launchers import bptt_launcher
from rlkit.tf.policies.memory.lstm_memory_policy import (
    SeparateLstmLinearCell)


def main():
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-sl"

    # n_seeds = 10
    # mode = "ec2"
    exp_prefix = "paper-6-14-HL-sl-H25"

    H = 25
    # noinspection PyTypeChecker
    variant = dict(
        H=H,
        exp_prefix=exp_prefix,
        algo_params=dict(
            num_batches_per_epoch=100,
            num_epochs=30,
            learning_rate=1e-3,
            batch_size=1000,
            eval_num_episodes=64,
            lstm_state_size=10,
            # rnn_cell_class=LSTMCell,
            # rnn_cell_params=dict(
            #     use_peepholes=True,
            # ),
            rnn_cell_class=SeparateLstmLinearCell,
            rnn_cell_params=dict(
                use_peepholes=True,
                env_noise_std=0,
                memory_noise_std=0,
                output_nonlinearity=tf.nn.tanh,
                # output_nonlinearity=tf.nn.softmax,
                env_hidden_sizes=[],
                output_dim=1,
            ),
            softmax=False,
        ),
        version='Supervised Learning',
        env_class=HighLow,
        env_params=dict(
            horizon=H,
        )
        # env_class=OneCharMemory,
    )

    exp_id = -1
    for _ in range(n_seeds):
        seed = random.randint(0, 999999)
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
