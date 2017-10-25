import abc
from collections import OrderedDict

import numpy as np

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.misc import logger


class MultitaskEnv(object, metaclass=abc.ABCMeta):
    """
    An environment with a task that can be specified with a goal state.
    """

    def __init__(self):
        self.multitask_goal = np.zeros(self.goal_dim)
        self.goal_dim_weights = np.ones(self.goal_dim)

    def set_goal(self, goal):
        self.multitask_goal = goal

    @abc.abstractmethod
    def sample_goal_states(self, batch_size):
        pass

    @property
    @abc.abstractmethod
    def goal_dim(self) -> int:
        """
        :return: int, dimension of goal state
        """
        pass

    @staticmethod
    def print_goal_state_info(goal):
        """
        Used for debugging.
        """
        print("Goal = ", goal)

    @abc.abstractmethod
    def sample_actions(self, batch_size):
        pass

    @abc.abstractmethod
    def sample_states(self, batch_size):
        pass

    """
    Functions you probably don't need to override.
    """

    def sample_goal_state_for_rollout(self):
        """
        These goal states are fed to a policy when the policy wants to actually
        do rollouts.
        :return:
        """
        goal_state = self.sample_goal_states(1)[0]
        return self.modify_goal_state_for_rollout(goal_state)

    def convert_ob_to_goal_state(self, obs):
        """
        Convert a raw environment observation into a goal state (if possible).

        This observation should NOT include the goal state.
        """
        if isinstance(obs, np.ndarray):
            return self.convert_obs_to_goal_states(
                np.expand_dims(obs, 0)
            )[0]
        else:
            return self.convert_obs_to_goal_states_pytorch(
                obs.unsqueeze(0)
            )[0]

    """
    Check out these default functions below! You may want to override them.
    """

    def compute_rewards(self, obs, action, next_obs, goal_states):
        return - np.linalg.norm(
            self.convert_obs_to_goal_states(next_obs) - goal_states,
            axis=1,
            keepdims=True,
            ord=1,
        )

    def convert_obs_to_goal_states(self, obs):
        """
        Convert a raw environment observation into a goal state (if possible).

        This observation should NOT include the goal state.
        """
        return obs

    def convert_obs_to_goal_states_pytorch(self, obs):
        """
        PyTorch version of `convert_obs_to_goal_state`.
        """
        return self.convert_obs_to_goal_states(obs)

    def modify_goal_state_for_rollout(self, goal_state):
        """
        Modify a goal state so that it's appropriate for doing a rollout.

        Common use case: zero out the goal velocities.
        :param goal_state:
        :return:
        """
        return goal_state

    def sample_irrelevant_goal_dimensions(self, goal, batch_size):
        """
        Copy the goal a bunch of time, but replace irrelevant goal dimensions
        with sampled values.

        For example, if you care about the position but not about the velocity,
        copy the velocity `batch_size` number of times, and then sample a bunch
        of velocity values.

        This default implementation assumes every dimension in the goal state
        is important.

        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        raise NotImplementedError("Nothing should be using this")
        # return np.repeat(
        #     np.expand_dims(goal, 0),
        #     batch_size,
        #     axis=0
        # )

    def sample_dimensions_irrelevant_to_oc(self, goal, obs, batch_size):
        """
        Copy the goal a bunch of time, but replace irrelevant goal dimensions
        with sampled values.

        For example, if you care about the position but not about the velocity,
        copy the velocity `batch_size` number of times, and then sample a bunch
        of velocity values.

        This default implementation assumes every dimension in the goal state
        is important.

        :param goal: np.ndarray, shape GOAL_DIM
        :param batch_size:
        :return: ndarray, shape SAMPLE_SIZE x GOAL_DIM
        """
        raise NotImplementedError()
        # return np.repeat(
        #     np.expand_dims(goal, 0),
        #     batch_size,
        #     axis=0
        # )


    def log_diagnostics(self, paths):
        statistics = OrderedDict()

        observations = np.vstack([path['observations'] for path in paths])
        goal_states = np.vstack([path['goal_states'] for path in paths])
        actions = np.vstack([path['actions'] for path in paths])
        final_distances = []
        for path in paths:
            reached = self.convert_ob_to_goal_state(path['observations'][-1])
            goal = path['goal_states'][-1]
            final_distances.append(
                np.linalg.norm(reached - goal)
            )
        final_distances = np.array(final_distances)

        goal_distances = np.linalg.norm(
            self.convert_obs_to_goal_states(observations) - goal_states,
            axis=1,
        )
        statistics.update(create_stats_ordered_dict(
            'Multitask distance to goal', goal_distances
        ))
        statistics.update(create_stats_ordered_dict(
            'Multitask final distance to goal', final_distances
        ))
        rewards = self.compute_rewards(
            observations[:-1, ...],
            actions[:-1, ...],
            observations[1:, ...],
            goal_states[:-1, ...],
        )
        statistics.update(create_stats_ordered_dict(
            'Multitask Env Rewards', rewards,
        ))
        for key, value in statistics.items():
            logger.record_tabular(key, value)
