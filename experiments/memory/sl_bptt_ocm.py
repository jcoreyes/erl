"""
Supervised learning BPTT on OCM.
"""
from itertools import product
import random
from railrl.launchers.rnn_launchers import bptt_launcher
from railrl.launchers.launcher_util import run_experiment
from tensorflow.contrib.rnn import (
    LayerNormBasicLSTMCell,
    LSTMCell,
    BasicLSTMCell,
)
from railrl.policies.memory.action_aware_memory_policy import DecoupledLSTM
from railrl.policies.memory.lstm_memory_policy import (
    LstmLinearCell,
)




def main():
    num_seeds = 5
    num_values = 2
    version = 'supervised_learning'
    exp_prefix = '4-30-comparison-short'
    for H, (rnn_cell_class, softmax), use_peepholes in product(
        [8, 16],
        [
            # (LayerNormBasicLSTMCell, False),
            # (LSTMCell, True),
            # (DecoupledLSTM, False),
            (LstmLinearCell, False),
        ],
        [True],
        # [False],
        # [None],
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
                # mode='ec2',
            )


if __name__ == "__main__":
    main()
