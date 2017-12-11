"""
Have a separate function so that if other code needs to merge/unmerge obs,
goals, and whatnot, they do it in the same way.
"""
import numpy as np


def merge_into_flat_obs(obs, goals, num_steps_left):
    return np.hstack((obs, goals, num_steps_left))


def extract_goals(flat_obs, ob_dim, goal_dim):
    return flat_obs[:, ob_dim:ob_dim+goal_dim]


def split_tau(flat_obs):
    return flat_obs[:, :-1], flat_obs[:, -1:]
