from collections import OrderedDict

import numpy as np

from railrl.data_management.env_replay_buffer import EnvReplayBuffer
from railrl.data_management.simple_replay_buffer import SimpleReplayBuffer

class TrajectoryReplayBuffer(SimpleReplayBuffer):
    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        stack_obs=1,
    ):
        super().__init__(
            max_replay_buffer_size,
            observation_dim * stack_obs,
            action_dim,
            env_info_sizes,
        )
        self.single_observation_dimension = observation_dim
        self.stack_obs = stack_obs

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by railrl.samplers.util.rollout
        """

        current_obs = np.zeros((self.stack_obs + 1, self.single_observation_dimension))

        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                agent_info,
                env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            path["agent_infos"],
            path["env_infos"],
        )):
            if i == 0:
                current_obs[-2, :] = obs
                current_obs[-1, :] = next_obs
            else:
                current_obs = np.vstack((
                    current_obs[1:, :],
                    next_obs
                ))
                assert (current_obs[-2, :] == obs).all(), "mismatch between obs and next_obs"
            obs1 = current_obs[:self.stack_obs, :].flatten()
            obs2 = current_obs[1:, :].flatten()
            self.add_sample(
                observation=obs1,
                action=action,
                reward=reward,
                next_observation=obs2,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()
