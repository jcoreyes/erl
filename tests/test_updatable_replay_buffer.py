import unittest

import numpy as np

from railrl.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from railrl.data_management.updatable_subtraj_replay_buffer import (
    UpdatableSubtrajReplayBuffer
)
from railrl.testing.tf_test_case import TFTestCase
from railrl.testing.stub_classes import StubEnv

def rand():
    return np.random.rand(1, 1)

class TestSubtrajReplayBuffer(TFTestCase):
    def test_size_add_none(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_pool_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        self.assertEqual(buff.num_can_sample(return_all=True), 0)

    def test_size_add_one(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_pool_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        observation = np.array([[0.5]]), np.array([[0.5]])
        action = np.array([0.5]), np.array([0.5])
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_can_sample(return_all=True), 0)

    def test_random_subtraj_shape(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_pool_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        observation = np.array([[0.5]]), np.array([[0.5]])
        action = np.array([0.5]), np.array([0.5])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        # First trajectory always goes in validation set
        subtrajs, _ = buff.random_subtrajectories(5, validation=True)
        self.assertEqual(subtrajs['env_obs'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['env_actions'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['next_env_obs'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['memories'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['next_memories'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['writes'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['rewards'].shape, (5, 2))
        self.assertEqual(subtrajs['terminals'].shape, (5, 2))

    def test_next_memory_equals_write(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_pool_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(10):
            observation = rand(), last_write
            write = rand()
            action = np.random.rand(1, 1), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        # First trajectory always goes in validation set
        subtrajs, _ = buff.random_subtrajectories(5, validation=True)
        self.assertNpEqual(subtrajs['next_memories'], subtrajs['writes'])

    def test_next_memory_equals_write_after_overflow(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_pool_size=10,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(13):
            observation = rand(), last_write
            write = rand()
            action = np.random.rand(1, 1), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        # First trajectory always goes in validation set
        subtrajs, _ = buff.random_subtrajectories(5, validation=True)
        self.assertNpEqual(subtrajs['next_memories'], subtrajs['writes'])



class StubMemoryEnv(ContinuousMemoryAugmented):
    def __init__(self):
        super().__init__(StubEnv(), 1)

if __name__ == '__main__':
    unittest.main()
