import abc


class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """
    A class used to save and replay data.
    """

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, terminal, **kwargs):
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
    def terminate_episode(self, terminal_observation, terminal, **kwargs):
        """
        Terminate the episode. The only reason this is needed in addition to
        add_sample is that sometimes you may want to terminate an episode
        prematurely, but without wanting it to look like a terminal state to
        the learning algorithm. For example, you might want to reset your
        environment every T time steps, but it's not like the T'th state is
        really a terminal state.

        :param terminal_observation: The last observation seen.
        :param terminal: Did the environment actually terminate by itself?
        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        :param path: Dict like on outputted by railrl.samplers.util.rollout
        """
        for (
                reward,
                terminal,
                action,
                obs,
                agent_info,
                env_info
        ) in zip(
            path["rewards"].reshape(-1, 1),
            path["terminals"].reshape(-1, 1),
            path["actions"],
            path["observations"],
            path["agent_infos"],
            path["env_infos"],
        ):
            self.add_sample(
                obs,
                action,
                reward,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode(
            path["final_observation"],
            path["terminals"][-1],
            agent_info=path["agent_infos"][-1],
            env_info=path["env_infos"][-1],
        )
