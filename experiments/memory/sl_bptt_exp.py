"""
Supervised learning BPTT on OCM.
"""
from itertools import product

from rlkit.envs.memory.high_low import HighLow
from rlkit.launchers.launcher_util import (
    run_experiment,
    set_seed,
)
from rlkit.launchers.rnn_launchers import bptt_launcher
from rlkit.tf.policies.memory.lstm_memory_policy import (
    SeparateRWALinearCell,
)


def main():
    n_seeds = 1
    mode = "here"
    exp_prefix = "dev-sl"

    # n_seeds = 10
    # mode = "ec2"
    # exp_prefix = "6-2-sl-rwa-vs-lstm"

    env_noise_std = 0
    memory_noise_std = 0
    for rnn_cell_class, H in product(
        [SeparateRWALinearCell],
        [512],
        # [RWACell, LSTMCell, GRUCell],
        # [512, 256, 128, 64],
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
                    # use_peepholes=True,
                    state_is_flat_externally=False,
                    output_dim=1,
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
