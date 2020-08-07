import unittest

import numpy as np

from rlkit.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented,
)
from rlkit.envs.memory.one_char_memory import OneCharMemory
from rlkit.testing.np_test_case import NPTestCase


class TestContinuousMemoryAugmented(NPTestCase):
    def test_dim_correct(self):
        ocm = OneCharMemory(n=5, num_steps=100)
        env = ContinuousMemoryAugmented(ocm, num_memory_states=10)
        self.assertEqual(env.action_space.flat_dim, 16)

    def test_memory_action_saved(self):
        ocm = OneCharMemory(n=5, num_steps=100)
        env = ContinuousMemoryAugmented(ocm, num_memory_states=10)
        env.reset()
        env_action = np.zeros(6)
        env_action[0] = 1
        memory_written = np.random.rand(10)
        action = [env_action, memory_written]
        _, saved_memory = env.step(action)[0]

        self.assertNpArraysEqual(memory_written, saved_memory)


if __name__ == '__main__':
    unittest.main()
