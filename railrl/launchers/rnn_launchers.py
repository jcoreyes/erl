"""
Various launchers for recurrent algorithms.
"""


def bptt_launcher(variant):
    """
    Run the BPTT algorithm.

    :param variant: Variant (dictionary) for experiment
    """
    from railrl.launchers.launcher_util import (
        set_seed,
    )
    from railrl.tf.bptt import Bptt
    H = variant['H']
    seed = variant['seed']
    env_class = variant['env_class']
    env_params = variant['env_params']
    set_seed(seed)

    env = env_class(**env_params)
    algorithm = Bptt(env, **variant['algo_params'])
    algorithm.train()
