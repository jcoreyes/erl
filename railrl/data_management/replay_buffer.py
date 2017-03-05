import abc


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, terminal):
        """
        Add a (state, observation, reward, terminal) tuple.

        :param observation:
        :param action:
        :param reward:
        :param terminal:
        :return:
        """
        pass

    @abc.abstractmethod
    def terminate_episode(self, terminal_observation):
        """
        Terminate the episode. The only reason this is needed in addition to
        add_sample is that sometimes you may want to terminate an episode
        prematurely, but without wanting it to look like a terminal state to
        the learning algorithm. For example, you might want to reset your
        environment every T time steps, but it's not like the T'th state is
        really a terminal state.

        :param terminal_observation: The last observation seen.
        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def size(self):
        """
        How many observation time steps are currently saved in this replay
        buffer, *including* terminal observations.
        :return:
        """
        pass

    # @abc.abstractmethod
    # def random_batch(self, batch_size):
    #     """
    #     Sample a random batch from this replay buffer.

    #     :param batch_size:
    #     :return: Dictionary with the following keys:
    #         - observations
    #         - actions
    #         - rewards
    #         - terminals
    #         - next_observations
    #     """
    #     pass
