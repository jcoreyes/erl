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
        Return # of transitions steps are currently saved in this replay buffer,
        i.e. how many valid (o_t, a_t, r_t, o_{t+1}) tuples there are.
        :return:
        """
        pass
