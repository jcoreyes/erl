import abc
from collections import OrderedDict

import numpy as np
from gym.spaces import Box

from railrl.envs.wrappers import ProxyEnv
from railrl.misc.eval_util import create_stats_ordered_dict
from railrl.core.serializable import Serializable
from railrl.core import logger as default_logger


def is_rllab_style_paths(paths):
    return "next_observations" not in paths[0]


class MultitaskEnv(object, metaclass=abc.ABCMeta):
    """
    An environment with a task that can be specified with a goal.
    Two big things:
    1. The goal should *not* be part of the state.
    2. Calls to reset() should *not* change the goal

    To change the goal, you need to explicitly call
    ```
    goal = env.sample_goal_for_rollout()
    env.set_goal(goal)
    env.reset()  # optional, but probably for the best
    ```

    If you want to append the goal to the state, do this:
    ```
    env = MyMultitaskEnv()
    env = MultitaskToFlatEnv(env)
    ```
    The above code will also make the goal change at every time step.
    See MultitaskToFlatEnv for more detail.

    If you want to change the goal at every call to reset(), but you do not
    want the goal to be appended to the state, do this:
    ```
    env = MyMultitaskEnv()
    env = MultitaskEnvToSilentMultitaskEnv(env)
    ```
    See `MultitaskEnvToSilentMultitaskEnv` for more detail.
    """

    def __init__(self, distance_metric_order=1, goal_dim_weights=None):
        self.multitask_goal = np.zeros(self.goal_dim)
        if goal_dim_weights is None:
            self.goal_dim_weights = np.ones(self.goal_dim)
        else:
            self.goal_dim_weights = np.array(goal_dim_weights)
        self.distance_metric_order = distance_metric_order

    @property
    @abc.abstractmethod
    def goal_dim(self) -> int:
        """
        :return: int, dimension of goal vector
        """
        pass

    @abc.abstractmethod
    def sample_goals(self, batch_size):
        pass

    @abc.abstractmethod
    def convert_obs_to_goals(self, obs):
        """
        Convert a raw environment observation into a goal state (if possible).
        """
        pass

    """
    Functions you probably don't need to override.
    """
    def oc_reward(
            self, predicted_states, goals, current_states
    ):
        return self.oc_reward_on_goals(
            self.convert_obs_to_goals(predicted_states),
            goals,
            current_states
        )

    def sample_goal_for_rollout(self):
        """
        These goal states are fed to a policy when the policy wants to actually
        do rollouts.
        :return:
        """
        goal = self.sample_goals(1)[0]
        return self.modify_goal_for_rollout(goal)

    def convert_ob_to_goal(self, obs):
        """
        Convert a raw environment observation into a goal state (if possible).

        This observation should NOT include the goal state.
        """
        if isinstance(obs, np.ndarray):
            return self.convert_obs_to_goals(
                np.expand_dims(obs, 0)
            )[0]
        else:
            return self.convert_obs_to_goals_pytorch(
                obs.unsqueeze(0)
            )[0]

    def compute_reward(self, ob, action, next_ob, goal):
        return self.compute_rewards(
            ob[None], action[None], next_ob[None], goal[None]
        )

    """
    Check out these default functions below! You may want to override them.
    """
    def set_goal(self, goal):
        self.multitask_goal = goal

    def compute_rewards(self, obs, actions, next_obs, goals):
        return - np.linalg.norm(
            self.convert_obs_to_goals(next_obs) - goals,
            axis=1,
            keepdims=True,
            ord=self.distance_metric_order,
        )

    def convert_obs_to_goals_pytorch(self, obs):
        """
        PyTorch version of `convert_obs_to_goals`.
        """
        return self.convert_obs_to_goals(obs)

    def modify_goal_for_rollout(self, goal):
        """
        Modify a goal state so that it's appropriate for doing a rollout.

        Common use case: zero out the goal velocities.
        :param goal:
        :return:
        """
        return goal

    def log_diagnostics(self, paths, logger=default_logger):
        list_of_goals = extract_list_of_goals(paths)
        if list_of_goals is None:
            return
        final_differences = []
        for path, goals in zip(paths, list_of_goals):
            reached = self.convert_ob_to_goal(path['observations'][-1])
            final_differences.append(reached - goals[-1])

        statistics = OrderedDict()

        if is_rllab_style_paths(paths):
            observations = np.vstack([
                path["observations"][:-1] for path in paths
            ])
            next_observations = np.vstack([
                path["observations"][1:] for path in paths
            ])
            actions = np.vstack([
                path["actions"][:-1] for path in paths
            ])
            goals = np.vstack([goals[:-1] for goals in list_of_goals])
        else:
            goals = np.vstack(list_of_goals)
            observations = np.vstack([path['observations'] for path in paths])
            next_observations = np.vstack([path['next_observations'] for path in paths])
            actions = np.vstack([path['actions'] for path in paths])
        for order in [1, 2]:
            final_distances = np.linalg.norm(
                np.array(final_differences),
                axis=1,
                ord=order,
            )
            goal_distances = np.linalg.norm(
                self.convert_obs_to_goals(observations) - np.vstack(goals),
                axis=1,
                ord=order,
            )
            statistics.update(create_stats_ordered_dict(
                'Multitask L{} distance to goal'.format(order),
                goal_distances,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Multitask Final L{} distance to goal'.format(order),
                final_distances,
                always_show_all_stats=True,
            ))
        rewards = self.compute_rewards(
            observations,
            actions,
            next_observations,
            goals,
        )
        statistics.update(create_stats_ordered_dict(
            'Multitask Env Rewards', rewards,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)

    """
    Optional functions to implement, since most of my code doesn't use these
    any more.
    """
    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        """
        Copy the goal a bunch of time, but replace irrelevant goal dimensions
        with sampled values.

        For example, if you care about the position but not about the velocity,
        copy the velocity `batch_size` number of times, and then sample a bunch
        of velocity values.

        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        pass

    def sample_actions(self, batch_size):
        pass

    def sample_states(self, batch_size):
        pass

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        """
        Create the OC goal state a bunch of time, but replace irrelevant goal
        dimensions with sampled values.

        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape `batch_size` x GOAL_DIM
        """
        pass

    def oc_reward_on_goals(
            self, predicted_goals, goals, current_states
    ):
        pass

    def cost_fn(self, states, actions, next_states):
        """
        This is added for model-based code. This is COST not reward.
        So lower is better.

        :param states:  (BATCH_SIZE x state_dim) numpy array
        :param actions:  (BATCH_SIZE x action_dim) numpy array
        :param next_states:  (BATCH_SIZE x state_dim) numpy array
        :return: (BATCH_SIZE, ) numpy array
        """
        if len(next_states.shape) == 1:
            next_states = np.expand_dims(next_states, 0)
        actual = self.convert_obs_to_goals(next_states)
        desired = self.multitask_goal * np.ones_like(actual)
        diff = actual - desired
        diff *= self.goal_dim_weights
        costs = np.linalg.norm(
            diff,
            axis=1,
            ord=1,
        )
        return costs


def extract_list_of_goals(paths):
    """
    Return list of goals. Each element in list is an array of goals and
    correspond to the goal from different paths.

    Returns None if it's not possible to extract goals from the paths.
    :param paths:
    :return:
    """
    if len(paths) == 0:
        return None

    if 'goals' in paths[0]:
        return [path['goals'] for path in paths]

    if 'env_infos' in paths[0]:
        env_infos = paths[0]['env_infos']
        if isinstance(env_infos, dict):  # rllab style paths
            return [path['env_infos']['goal'] for path in paths]
        elif 'goal' in env_infos[0]:
            return [
                [info['goal'] for info in path['env_infos']]
                for path in paths
            ]
    return None


class MultitaskToFlatEnv(ProxyEnv, Serializable):
    """
    This environment tasks a multitask environment and appends the goal to
    the state.
    """
    def __init__(
            self,
            env: MultitaskEnv,
            give_goal_difference=False,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        self._wrapped_obs_dim = env.observation_space.low.size
        self.give_goal_difference = give_goal_difference
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        wrapped_low = self.observation_space.low
        low = np.hstack((
            wrapped_low,
            min(wrapped_low) * np.ones(self._wrapped_env.goal_dim)
        ))
        wrapped_high = self.observation_space.low
        high = np.hstack((
            wrapped_high,
            max(wrapped_high) * np.ones(self._wrapped_env.goal_dim)
        ))
        self.observation_space = Box(low, high)

    def step(self, action):
        ob, reward, done, info_dict = self._wrapped_env.step(action)
        new_ob = self._add_goal_to_observation(ob)
        return new_ob, reward, done, info_dict

    def reset(self):
        self._wrapped_env.set_goal(self._wrapped_env.sample_goal_for_rollout())
        ob = super().reset()
        new_ob = self._add_goal_to_observation(ob)
        return new_ob

    def _add_goal_to_observation(self, ob):
        if self.give_goal_difference:
            goal_difference = (
                self._wrapped_env.multitask_goal
                - self._wrapped_env.convert_ob_to_goal(ob)
            )
            return np.hstack((ob, goal_difference))
        else:
            return np.hstack((ob, self._wrapped_env.multitask_goal))

    def cost_fn(self, states, actions, next_states):
        if len(next_states.shape) == 1:
            states = states[None]
            actions = actions[None]
            next_states = next_states[None]
        unwrapped_states = states[:, :self._wrapped_obs_dim]
        unwrapped_next_states = next_states[:, :self._wrapped_obs_dim]
        return self._wrapped_env.cost_fn(
            unwrapped_states,
            actions,
            unwrapped_next_states,
        )


class MultitaskEnvToSilentMultitaskEnv(ProxyEnv, Serializable):
    """
    Normally, reset() on a multitask env doesn't change the goal.
    Now, reset will silently change the goal.
    """
    def reset(self):
        self._wrapped_env.set_goal(self._wrapped_env.sample_goal_for_rollout())
        return super().reset()

    def cost_fn(self, states, actions, next_states):
        return self._wrapped_env.cost_fn(
            states,
            actions,
            next_states,
        )

    def sample_goal_for_rollout(self):
        return self._wrapped_env.sample_goal_for_rollout()

    def sample_goals(self, batch_size):
        return self._wrapped_env.sample_goals(batch_size)

    def sample_states(self, batch_size):
        return self._wrapped_env.sample_states(batch_size)
