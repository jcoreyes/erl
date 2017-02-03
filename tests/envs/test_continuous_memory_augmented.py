import unittest
import numpy as np
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented,
)
from railrl.misc.np_test_case import NPTestCase


class TestContinuousMemoryAugmented(NPTestCase):
    def test_dim_correct(self):
        ocm = OneCharMemory(n=5, num_steps=100)
        env = ContinuousMemoryAugmented(ocm, num_memory_states=10)
        self.assertEqual(env.action_space.flat_dim, 16)


if __name__ == '__main__':
    unittest.main()
