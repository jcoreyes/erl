"""
Supervised learning with full BPTT.
"""
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

from railrl.envs.memory.high_low import HighLow
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from railrl.policies.memory.lstm_memory_policy import (
    SeparateLstmLinearCell)


def run_sl_exp(variant):
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.algos.bptt import Bptt
    H = variant['H']
    seed = variant['seed']
    env_class = variant['env_class']
    set_seed(seed)

    env = env_class(num_steps=H)
    algorithm = Bptt(env, **variant['algo_params'])
    algorithm.train()


def main():
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-sl"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "5-27-benchmark-sl-ocm-sweep-h"

    # noinspection PyTypeChecker
    variant = dict(
        H=128,
        exp_prefix=exp_prefix,
        algo_params=dict(
            num_batches_per_epoch=10000//32,
            num_epochs=100,
            learning_rate=1e-3,
            batch_size=32,
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
            ),
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
            run_sl_exp,
            exp_prefix=exp_prefix,
            seed=seed,
            mode=mode,
            variant=variant,
            exp_id=exp_id,
        )


if __name__ == "__main__":
    main()
