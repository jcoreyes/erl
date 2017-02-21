import unittest

import numpy as np
import tensorflow as tf

from railrl.data_management.episode_replay_buffer import EpisodeReplayBuffer
from railrl.misc.tf_test_case import TFTestCase
from rllab.spaces.box import Box


class TestEpisodeReplayBuffer(TFTestCase):
    def test_size_add_none(self):
        env = StubEnv()
        buff = EpisodeReplayBuffer(
            3,
            env,
        )
        self.assertEqual(buff.size, 0)

    def test_size_add_one(self):
        env = StubEnv()
        buff = EpisodeReplayBuffer(
            3,
            env,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.size, 1)

    def test_size_add_many(self):
        env = StubEnv()
        buff = EpisodeReplayBuffer(
            3,
            env,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        self.assertEqual(buff.size, 10)

    def test_size_after_episode_terminated(self):
        env = StubEnv()
        buff = EpisodeReplayBuffer(
            3,
            env,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        buff.terminate_episode(observation)
        self.assertEqual(buff.size, 11)

    def test_get_random_subtraj_batch_size(self):
        env = StubEnv()
        buff = EpisodeReplayBuffer(
            3,
            env,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        subtrajs = buff.random_subtrajectories(5, 2)
        for time_values in subtrajs.values():
            self.assertEqual(len(time_values), 5)

    def test_get_random_subtraj_indiv_sizes(self):
        env = StubEnv()
        buff = EpisodeReplayBuffer(
            3,
            env,
        )
        observation = np.array([[0.5]])
        action = np.array([[0.5]])
        for _ in range(10):
            buff.add_sample(observation, action, 1, False)
        subtrajs = buff.random_subtrajectories(5, 2)
        for time_values in subtrajs.values():
            for value in time_values:
                self.assertEqual(len(value), 2)


class StubEnv(object):
    def __init__(self):
        low = np.array([0.])
        high = np.array([1.])
        self.action_space = Box(low, high)
        self.observation_space = Box(low, high)


if __name__ == '__main__':
    unittest.main()
