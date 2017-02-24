"""
Various launchers for recurrent algorithms.
"""

def bptt_launcher(variant):
    """
    Run a simple LSTM on an environment.

    :param variant: Dictionary of dictionary with the following keys:
        - algo_params
        - env_params
        - qf_params
        - policy_params
    :return:
    """
    from railrl.algos.bptt import Bptt
    from railrl.launchers.launcher_util import get_env_settings
    env_settings = get_env_settings(**variant['env_params'])
    env = env_settings['env']
    algorithm = Bptt(env)
    algorithm.train()
