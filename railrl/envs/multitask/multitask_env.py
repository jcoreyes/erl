import abc
from collections import OrderedDict

import numpy as np

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.misc import logger as rllab_logger
from rllab.spaces import Box


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

    def __init__(self, distance_metric_order=1):
        self.multitask_goal = np.zeros(self.goal_dim)
        self.goal_dim_weights = np.ones(self.goal_dim)
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

    """
    Check out these default functions below! You may want to override them.
    """
    def set_goal(self, goal):
        self.multitask_goal = goal

    def compute_rewards(self, obs, action, next_obs, goals):
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

    def log_diagnostics(self, paths, logger=rllab_logger):
        if 'goals' not in paths[0]:
            return
        statistics = OrderedDict()

        observations = np.vstack([path['observations'] for path in paths])
        next_observations = np.vstack([path['next_observations'] for path in paths])
        goals = np.vstack([path['goals'] for path in paths])
        actions = np.vstack([path['actions'] for path in paths])
        final_differences = []
        for path in paths:
            reached = self.convert_ob_to_goal(path['observations'][-1])
            goal = path['goals'][-1]
            final_differences.append(reached - goal)
        for order in [1, 2]:
            final_distances = np.linalg.norm(
                np.array(final_differences),
                axis=1,
                ord=order,
            )
            goal_distances = np.linalg.norm(
                self.convert_obs_to_goals(observations) - goals,
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
        This is added for Abhishek's model-based code.
        """
        if len(next_states.shape) == 1:
            next_states = np.expand_dims(next_states, 0)
        actual = self.convert_obs_to_goals(next_states)
        desired = self.multitask_goal * np.ones_like(actual)
        costs = np.linalg.norm(
            actual - desired,
            axis=1,
            ord=1,
        )
        return costs


class MultitaskToFlatEnv(ProxyEnv, Serializable):
    """
    This environment tasks a multitask environment and appends the goal to
    the state.
    """
    def __init__(
            self,
            env: MultitaskEnv,
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
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @property
    def observation_space(self):
        wrapped_low = super().observation_space.low
        low = np.hstack((
            wrapped_low,
            min(wrapped_low) * np.ones(self._wrapped_env.goal_dim)
        ))
        wrapped_high = super().observation_space.low
        high = np.hstack((
            wrapped_high,
            max(wrapped_high) * np.ones(self._wrapped_env.goal_dim)
        ))
        return Box(low, high)

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
        goal_difference = (
            self._wrapped_env.multitask_goal
            - self._wrapped_env.convert_ob_to_goal(ob)
        )
        return np.hstack((ob, goal_difference))

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
