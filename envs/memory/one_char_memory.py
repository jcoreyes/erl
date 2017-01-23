from random import randint
from rllab.envs.base import Env
from rllab.spaces.discrete import Discrete


class OneCharMemory(Env):
    """
    A simple env whose output is a value `X` the first time step, followed by a
    fixed number of zeros.

    There are `n` different values that `X` can take on (excluding 0).

    The reward is zero for all steps except the end, where the agent gets a
    reward of 1 if `X` is returned correctly.


    Both the actions and observations are represented as one-hot vectors.
    """
    def __init__(self, n=4, num_steps=1000):
        """
        :param n: Number of different values that could be returned
        :param num_steps: How many steps the agent needs to remember.
        """
        super().__init__()
        self.num_steps = num_steps
        self.n = n
        self._action_space = Discrete(self.n + 1)
        self._observation_space = Discrete(self.n + 1)
        self._t = 1

        self._target = None
        self._next_obs = None

    def step(self, action):
        observation = self._observation_space.flatten(self._next_obs)
        self._next_obs = 0

        done = self._t == self.num_steps
        self._t += 1

        if done:
            reward = int(
                self._observation_space.unflatten(action) == self._target
            )
        else:
            reward = 0
        info = {'target': self.n}
        return observation, reward, done, info

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    def reset(self):
        self._target = randint(1, self.n)
        self._next_obs = self._target
        self._t = 0

    @property
    def observation_space(self):
        return self._observation_space
