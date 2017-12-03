import numpy as np


def merge_into_flat_obs(obs, goals, num_steps_left):
    # Have a separate function so that if other code needs to merge obs,
    # goals, and whatnot, it does it in the same way.
    return np.hstack((obs, goals, num_steps_left))


def split_tau(flat_obs):
    return flat_obs[:, :-1], flat_obs[:, -1:]
