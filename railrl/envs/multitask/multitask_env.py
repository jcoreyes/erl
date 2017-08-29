import abc
import numpy as np


class MultitaskEnv(object, metaclass=abc.ABCMeta):
    """
    An environment with a task that can be specified with a goal state.
    """
    @abc.abstractmethod
    def set_goal(self, goal):
        pass

    @abc.abstractmethod
    def sample_goal_states(self, batch_size):
        pass

    @abc.abstractmethod
    def sample_goal_states_for_rollouts(self, batch_size):
        """
        These goal states are fed to a policy when the policy wants to actually
        do rollouts.
        :param batch_size:
        :return:
        """
        pass

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
        return np.repeat(
            np.expand_dims(goal, 0),
            batch_size,
            axis=0
        )

    @abc.abstractmethod
    def sample_actions(self, batch_size):
        pass

    @abc.abstractmethod
    def sample_states(self, batch_size):
        pass

    def compute_rewards(self, obs, action, next_obs, goal_states):
        return - np.linalg.norm(
            self.convert_obs_to_goal_states(next_obs) - goal_states,
            axis=1,
        )

    @abc.abstractmethod
    def convert_obs_to_goal_states(self, obs):
        """
        Convert a raw environment observation into a goal state (if possible).

        This observation should NOT include the goal state.
        """
        pass

    def convert_obs_to_goal_states_pytorch(self, obs):
        """
        PyTorch version of `convert_obs_to_goal_state`.
        """
        return self.convert_obs_to_goal_states(obs)

    @property
    @abc.abstractmethod
    def goal_dim(self):
        """
        :return: int, dimension of goal state
        """
        pass

    @staticmethod
    def print_goal_state_info(goal):
        """
        Used for debugging.
        """
        pass
