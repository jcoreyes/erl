"""
Supervised learning BPTT on OCM.
"""
from railrl.launchers.rnn_launchers import bptt_launcher
from railrl.launchers.launcher_util import run_experiment


def main():
    variant = dict(
        env_params=dict(
            env_id='ocm',
        ),
    )
    seed = 0
    run_experiment(
        bptt_launcher,
        exp_prefix="dev-bptt",
        seed=seed,
        variant=variant,
    )


if __name__ == "__main__":
    main()
