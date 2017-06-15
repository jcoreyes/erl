import numpy as np
from rllab.spaces.product import Product

# TODO(vpong): unittest this
def split_paths(paths):
    """
    Split paths from rllab's rollout function into rewards, terminals, obs
    actions, and next_obs
    terminals
    :param paths:
    :return:
    """
    rewards = [path["rewards"] for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = []
    terminals = []
    for path in paths:
        next_obs_i = path["observations"][1:, :]
        next_obs_i = np.vstack((next_obs_i,
                                np.zeros_like(path["observations"][0:1])))
        next_obs.append(next_obs_i)

        terminal_i = np.zeros_like(path["rewards"])
        terminal_i[-1] = 1
        terminals.append(terminal_i)
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    return rewards, terminals, obs, actions, next_obs


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
