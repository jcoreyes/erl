import numpy as np
from sklearn.metrics import log_loss
from random import randint
from rllab.envs.base import Env
from rllab.misc import special2 as special
from rllab.spaces.box import Box
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

    def __init__(self, n=4, num_steps=100, reward_for_remembering=1000):
        """
        :param n: Number of different values that could be returned
        :param num_steps: How many steps the agent needs to remember.
        :param reward_for_remembering: The reward for remembering the number.
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

        self._target_number = None
        self._next_obs_number = None

    def step(self, action):
        # flatten = to one hot...not sure why it was given that name.
        observation = self._get_next_observation()
        self._next_obs_number = 0

        done = self._t == self.num_steps
        self._t += 1

        if done:
            reward = -(log_loss(self._get_target_onehot(), action) *
                       self._reward_for_remembering)
        else:
            reward = -log_loss(self.zero, action)
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
            high=self.n,
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
