"""
Utility functions built around rllab functions/objects/classes.
"""
import os.path as osp
import joblib
import numpy as np

from rllab.spaces.product import Product
from rllab.misc import logger


def split_paths(paths):
    """
    Split paths from rllab's rollout function into rewards, terminals, obs
    actions, and next_obs
    terminals
    :param paths:
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
     - rewards
     - terminal
     - rewards
     - rewards
     - rewards
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = []
    for path in paths:
        next_obs_i = path["observations"][1:, :]
        next_obs_i = np.vstack((next_obs_i,
                                np.zeros_like(path["observations"][0:1])))
        next_obs.append(next_obs_i)
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs

def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_dict(paths, dict_name, scalar_name):
    return np.vstack([
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ])


def get_action_dim(**kwargs):
    env_spec = kwargs.get('env_spec', None)
    action_dim = kwargs.get('action_dim', None)
    assert env_spec or action_dim
    if action_dim:
        return action_dim

    if isinstance(env_spec.action_space, Product):
        return tuple(
            c.flat_dim for c in env_spec.action_space.components
        )
    else:
        return env_spec.action_space.flat_dim


def get_observation_dim(**kwargs):
    env_spec = kwargs.get('env_spec', None)
    observation_dim = kwargs.get('observation_dim', None)
    assert env_spec or observation_dim
    if observation_dim:
        return observation_dim

    if isinstance(env_spec.observation_space, Product):
        return tuple(
            c.flat_dim for c in env_spec.observation_space.components
        )
    else:
        return env_spec.observation_space.flat_dim


def split_flat_product_space_into_components_n(product_space, xs):
    """
    Split up a flattened block into its components

    :param product_space: ProductSpace instance
    :param xs: N x flat_dim
    :return: list of (N x component_dim)
    """
    dims = [c.flat_dim for c in product_space.components]
    return np.split(xs, np.cumsum(dims)[:-1], axis=-1)


def get_average_returns(paths):
    returns = [sum(path["rewards"]) for path in paths]
    return np.mean(returns)


"""
Logger Util
"""

def get_table_key_set(logger):
    return set(key for key, value in logger._tabular)


def get_logger_table_dict():
    return dict(logger._tabular)


def save_extra_data_to_snapshot_dir(data):
    """
    Save extra data to the snapshot dir.

    :param logger:
    :param data:
    :return:
    """
    file_name = osp.join(logger._snapshot_dir, 'extra_data.pkl')
    joblib.dump(data, file_name, compress=3)
