import unittest

import numpy as np

from rlkit.envs.memory.continuous_memory_augmented import (
    ContinuousMemoryAugmented
)
from rlkit.data_management.updatable_subtraj_replay_buffer import (
    UpdatableSubtrajReplayBuffer
)
from rlkit.testing.np_test_case import NPTestCase
from rlkit.testing.stub_classes import StubEnv


def rand(dim=1):
    return np.random.rand(1, dim)

class TestSubtrajReplayBuffer(NPTestCase):
    def test_size_add_none(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 0)

    def test_size_add_one(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        observation = rand(), rand()
        action = rand(), rand()
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.num_steps_can_sample(return_all=True), 0)

    def test_random_subtraj_shape(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        observation = rand(), rand()
        action = rand(), rand()
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        subtrajs, _ = buff.random_subtrajectories(5)
        self.assertEqual(subtrajs['env_obs'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['env_actions'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['next_env_obs'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['memories'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['next_memories'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['writes'].shape, (5, 2, 1))
        self.assertEqual(subtrajs['rewards'].shape, (5, 2))
        self.assertEqual(subtrajs['dloss_dwrites'].shape, (5, 2, 1))

    def test_next_memory_equals_write(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(10):
            observation = rand(), last_write
            write = rand()
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        subtrajs, _ = buff.random_subtrajectories(5)
        self.assertNpEqual(subtrajs['next_memories'], subtrajs['writes'])

    def test_next_memory_equals_write_after_overflow(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=10,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(13):
            observation = rand(), last_write
            write = rand()
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        subtrajs, _ = buff.random_subtrajectories(5)
        self.assertNpEqual(subtrajs['next_memories'], subtrajs['writes'])

    def test_dloss_dwrites_are_zero_initially(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(13):
            observation = rand(), last_write
            write = rand()
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        subtrajs, _ = buff.random_subtrajectories(5)
        self.assertNpEqual(subtrajs['dloss_dwrites'], np.zeros((5, 2, 1)))

    def test__fixed_start_indices(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        for _ in range(25):
            observation = rand(), rand()
            action = rand(), rand()
            buff.add_sample(observation, action, 1, False)
        _, start_indices = buff.random_subtrajectories(15)
        _, new_start_indices = buff.random_subtrajectories(
            15,
            _fixed_start_indices=start_indices,
        )
        self.assertNpEqual(start_indices, new_start_indices)

    def test_update_memories_updates_memories(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(13):
            observation = rand(), last_write
            write = rand()
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        # First trajectory always goes in validation set
        start_indices = [0, 4, 8]
        new_writes = np.random.rand(len(start_indices), 2, 1)
        buff.update_write_subtrajectories(new_writes, start_indices)
        new_subtrajs, _ = buff.random_subtrajectories(
            len(start_indices),
            _fixed_start_indices=start_indices,
        )
        self.assertNpEqual(new_subtrajs['writes'], new_writes)

    def test_update_memories_updates_memories_2d(self):
        env = StubMemoryEnv(2)
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=2,
        )
        last_write = rand(2)
        for _ in range(13):
            observation = rand(), last_write
            write = rand(2)
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        start_indices = [0, 4, 8]
        new_writes = np.random.rand(len(start_indices), 2, 2)
        buff.update_write_subtrajectories(new_writes, start_indices)
        new_subtrajs, _ = buff.random_subtrajectories(
            len(start_indices),
            _fixed_start_indices=start_indices,
        )
        self.assertNpEqual(new_subtrajs['writes'], new_writes)

    def test_update_memories_does_not_update_other_memories(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        buff.terminate_episode((rand(), rand()), True)
        for _ in range(5):
            observation = rand(), last_write
            write = rand()
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        # White box testing...sue me.
        old_memories = buff._memories.copy()
        """
        For writes
        0 - same
        1 - different
        2 - different
        3 - same
        4 - same

        For memories
        0 - same
        1 - same
        2 - different
        3 - different
        4 - same
        """
        start_indices = [1]
        written_writes = np.random.rand(len(start_indices), 2, 1)
        buff.update_write_subtrajectories(written_writes, start_indices)
        new_memories = buff._memories

        expected_new_memories = old_memories.copy()
        expected_new_memories[2:4] = written_writes
        self.assertNpArraysNotEqual(old_memories, expected_new_memories)
        self.assertNpArraysNotEqual(old_memories, new_memories)
        self.assertNpEqual(new_memories, expected_new_memories)

    def test_update_dloss_dmemories_works(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(13):
            observation = rand(), last_write
            write = rand()
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        """
        internal
        dL/dm idx   dL/dw idx   changed?
        0           n/a         different
        1           0           different
        2           1           same
        3           2           same
        4           3           different
        5           4           different
        6           5           same
        7           6           same
        8           7           different
        9           8           different
        10          9           same
        11          10          same
        12          11          same
        13          12          same
        """
        start_indices = [0, 4, 8]
        dloss_dmem = np.random.rand(len(start_indices), 2, 1)
        buff.update_dloss_dmemories_subtrajectories(dloss_dmem, start_indices)
        new_subtrajs, _ = buff.random_subtrajectories(
            len(start_indices),
            _fixed_start_indices=start_indices,
        )
        expected_dloss_dwrite = np.zeros_like(dloss_dmem)
        for i in range(len(start_indices)):
            expected_dloss_dwrite[i, 0, :] = dloss_dmem[i, 1, :]
        self.assertNpEqual(new_subtrajs['dloss_dwrites'], expected_dloss_dwrite)

    def test_update_dloss_dmemories_works_overlap(self):
        env = StubMemoryEnv()
        buff = UpdatableSubtrajReplayBuffer(
            max_replay_buffer_size=100,
            env=env,
            subtraj_length=2,
            memory_dim=1,
        )
        last_write = rand()
        for _ in range(13):
            observation = rand(), last_write
            write = rand()
            action = rand(), write
            last_write = write
            buff.add_sample(observation, action, 1, False)
        """
        internal
        dL/dm idx   dL/dw idx   changed?
        0           n/a         different
        1           0           different
        2           1           same
        3           2           different
        4           3           different
        5           4           same
        """
        start_indices = [0, 3]
        dloss_dmem = np.random.rand(len(start_indices), 2, 1)
        buff.update_dloss_dmemories_subtrajectories(dloss_dmem, start_indices)
        new_subtrajs, _ = buff.random_subtrajectories(
            len(start_indices),
            _fixed_start_indices=[0, 1, 2, 3],
        )
        expected_dloss_dwrite = np.zeros((4, 2, 1))
        expected_dloss_dwrite[0, 0, :] = dloss_dmem[0, 1, :]
        expected_dloss_dwrite[1, 1, :] = dloss_dmem[1, 0, :]
        expected_dloss_dwrite[2, 0, :] = dloss_dmem[1, 0, :]
        expected_dloss_dwrite[2, 1, :] = dloss_dmem[1, 1, :]
        expected_dloss_dwrite[3, 0, :] = dloss_dmem[1, 1, :]
        self.assertNpEqual(new_subtrajs['dloss_dwrites'], expected_dloss_dwrite)


class StubMemoryEnv(ContinuousMemoryAugmented):
    def __init__(self, num_memory_states=1):
        super().__init__(StubEnv(), num_memory_states=num_memory_states)


if __name__ == '__main__':
    unittest.main()
