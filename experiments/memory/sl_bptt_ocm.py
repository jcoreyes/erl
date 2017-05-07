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
    ResidualLstmLinearCell,
)




def main():
    num_seeds = 1
    num_values = 2
    use_peepholes = True
    softmax = False
    # version = 'supervised_learning'
    # exp_prefix = 'dev-sl'
    version = 'proper-layer-norm'
    exp_prefix = '5-7-sl-test-layer-norm'
    env_noise_std = 0
    memory_noise_std = 0
    for H, rnn_cell_class in product(
        [64],
        # [0., 0.1, 0.3, 1],
        # [0., 0.5, 1., 3.],
        [LstmLinearCell],
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
                num_epochs=10,
                learning_rate=1e-3,
                batch_size=32,
                eval_num_episodes=64,
                lstm_state_size=10,
                rnn_cell_class=rnn_cell_class,
                rnn_cell_params=dict(
                    use_peepholes=use_peepholes,
                    env_noise_std=env_noise_std,
                    memory_noise_std=memory_noise_std,
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
