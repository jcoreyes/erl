"""
Supervised learning BPTT on OCM.
"""
from itertools import product
import random
from railrl.launchers.rnn_launchers import bptt_launcher
from railrl.launchers.launcher_util import run_experiment


def main():
    num_seeds = 3
    for H, num_values in product(
        [2, 4, 8, 16],
        [2, 4, 8, 16]
    ):
        variant = dict(
            env_params=dict(
                env_id='ocm',
                init_env_params=dict(
                    num_steps=H,
                    n=num_values,
                )
            ),
            algo_params=dict(
                num_batches_per_epoch=1000,
                num_epochs=100,
                learning_rate=1e-3,
                batch_size=32,
                eval_num_episodes=64,
                lstm_state_size=10,
            )
        )
        for _ in range(num_seeds):
            seed = random.randint(0, 100000)
            run_experiment(
                bptt_launcher,
                exp_prefix="3-8-bptt-sl-benchmark",
                seed=seed,
                variant=variant,
            )


if __name__ == "__main__":
    main()
