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
