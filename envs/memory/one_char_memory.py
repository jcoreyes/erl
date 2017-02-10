import numpy as np
from sklearn.metrics import log_loss
from random import randint

from railrl.misc.np_util import np_print_options
from railrl.pythonplusplus import clip_magnitude
from rllab.envs.base import Env
from rllab.misc import special2 as special
from rllab.misc.overrides import overrides
from rllab.spaces.box import Box
from rllab.misc import logger
from railrl.envs.supervised_learning_env import SupervisedLearningEnv
from cached_property import cached_property


class OneCharMemory(Env, SupervisedLearningEnv):
    """
    A simple env whose output is a value `X` the first time step, followed by a
    fixed number of zeros.

    The goal of the agent is to output zero for all time steps, and then
    output `X` in the last time step.

    Both the actions and observations are represented as probability vectors.
    There are `n` different values that `X` can take on (excluding 0),
    so the probability vector's dimension is n+1.

    The reward is the negative cross-entropy loss between the target one-hot
    vector and the probability vector outputted by the agent. Furthermore, the
    reward for the last time step is multiplied by `reward_for_remember`.
    """

    def __init__(
            self,
            n=4,
            num_steps=10,
            reward_for_remembering=1,
            max_reward_magnitude=1,
    ):
        """
        :param n: Number of different values that could be returned
        :param num_steps: How many steps the agent needs to remember.
        :param reward_for_remembering: The reward bonus for remembering the
        number. This number is added to the usual reward if the correct
        number has the maximum probability.
        :param max_reward_magnitude: Clip the reward magnitude to this value.
        """
        super().__init__()
        self.num_steps = num_steps
        self.n = n
        self._onehot_size = n + 1
        self._action_space = Box(
            np.zeros(self._onehot_size),
            np.ones(self._onehot_size)
        )
        self._observation_space = self._action_space
        self._t = 1
        self._reward_for_remembering = reward_for_remembering
        self._max_reward_magnitude = max_reward_magnitude

        self._target_number = None
        self._next_obs_number = None

        # For rendering
        self._last_reward = None
        self._last_action = None
        self._last_t = None

    def step(self, action):
        action = action.flatten()
        observation = self._get_next_observation()
        self._next_obs_number = 0

        done = self._t == self.num_steps
        self._last_t = self._t
        self._t += 1

        try:
            if done:
                reward = -log_loss(self._get_target_onehot(), action)
                if np.argmax(action) == self._target_number:
                    reward += self._reward_for_remembering
            else:
                reward = -log_loss(self.zero, action)
            reward = clip_magnitude(reward, self._max_reward_magnitude)
        except ValueError as e:
            print(e)
            import ipdb
            ipdb.set_trace()
            raise e
        self._last_reward = reward
        self._last_action = action
        info = {'target': self.n}
        return observation, reward, done, info

    @cached_property
    def zero(self):
        z = np.zeros(self._onehot_size)
        z[0] = 1
        return z

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    def reset(self):
        self._target_number = randint(1, self.n)
        self._next_obs_number = self._target_number
        self._t = 1
        return self._get_next_observation()

    def _get_next_observation(self):
        return special.to_onehot(self._next_obs_number, self._onehot_size)

    def _get_target_onehot(self):
        return special.to_onehot(self._target_number, self._onehot_size)

    @property
    def observation_space(self):
        return self._observation_space

    def get_batch(self, batch_size):
        targets = np.random.randint(
            low=1,
            high=self.n+1,
            size=batch_size,
        )
        onehot_targets = special.to_onehot_n(targets, self.feature_dim)
        X = np.zeros((batch_size, self.sequence_length, self.feature_dim))
        X[:, :, 0] = 1  # make the target 0
        X[:, 0, :] = onehot_targets
        Y = np.zeros((batch_size, self.sequence_length, self.target_dim))
        Y[:, :, 0] = 1  # make the target 0
        Y[:, -1, :] = onehot_targets
        return X, Y

    @property
    def feature_dim(self):
        return self.n + 1

    @property
    def target_dim(self):
        return self.n + 1

    @property
    def sequence_length(self):
        return self.horizon

    @overrides
    def render(self):
        logger.push_prefix("OneCharMemory(n={0})\t".format(self._target_number))
        if self._last_action is None:
            logger.log("No action taken.")
        else:
            if self._last_t == 1:
                logger.log("--- New Episode ---")
            logger.push_prefix("t={0}\t".format(self._last_t))
            with np_print_options(precision=4, suppress=False):
                logger.log("Action: {0}".format(
                    self._last_action,
                ))
            logger.log("Reward: {0}".format(
                self._last_reward,
            ))
            logger.pop_prefix()
        logger.pop_prefix()
