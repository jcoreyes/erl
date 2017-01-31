import unittest
import numpy as np
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.misc.np_test_case import NPTestCase


class TestOneCharMemory(NPTestCase):
    def test_dim_correct(self):
        env = OneCharMemory(n=5, num_steps=100)
        self.assertEqual(env.feature_dim, 6)
        self.assertEqual(env.target_dim, 6)
        self.assertEqual(env.sequence_length, 100)

    def test_get_batch_shape(self):
        env = OneCharMemory(n=5, num_steps=100)
        X, Y = env.get_batch(batch_size=3)

        self.assertEqual(X.shape, (3, 100, 6))
        self.assertEqual(Y.shape, (3, 100, 6))

    def test_batch_x_first_and_y_last_are_equal(self):
        env = OneCharMemory(n=3, num_steps=4)
        X, Y = env.get_batch(batch_size=100)

        self.assertNpEqual(X[:, 0, :], Y[:, -1, :])

    def test_middle_of_batches_are_all_zero_onehots(self):
        env = OneCharMemory(n=3, num_steps=4)
        X, Y = env.get_batch(batch_size=100)

        self.assertNpArrayConstant(X[:, 1:, 1:], 0)
        self.assertNpArrayConstant(Y[:, :-1, 1:], 0)
        self.assertNpArrayConstant(X[:, 1:, 0], 1)
        self.assertNpArrayConstant(Y[:, :-1, 0], 1)

    def test_first_x_is_one_hot(self):
        env = OneCharMemory(n=3, num_steps=4)
        X, Y = env.get_batch(batch_size=100)

        self.assertNpArrayConstant(np.sum(np.array(1 == X[:, 0, :]), axis=1), 1)

    def test_init_state_is_one_hot(self):
        env = OneCharMemory(n=3, num_steps=4)
        init_state = env.reset()
        self.assertEqual(init_state.shape, (4,))

    def test_target_is_never_zero_one_hot(self):
        env = OneCharMemory(n=3, num_steps=4)
        X, Y = env.get_batch(batch_size=100)

        self.assertNpArrayConstant(np.sum(X[:, 0, 0]), 0)

    def test_reward_for_optimal_input_is_correct(self):
        env = OneCharMemory(n=3, num_steps=4, reward_for_remembering=10)
        init_obs = env.reset()

        action = np.zeros((4, 1))
        action[0] = 1
        for _ in range(3):
            next_ob, reward, terminal, _ = env.step(action)
            self.assertEqual(reward, 1)

        next_ob, reward, terminal, _ = env.step(init_obs)
        self.assertEqual(reward, 10)

    def test_reward_for_wrong_input_is_correct(self):
        env = OneCharMemory(n=3, num_steps=4, reward_for_remembering=10)
        init_obs = env.reset()

        action = init_obs
        for _ in range(3):
            next_ob, reward, terminal, _ = env.step(action)
            self.assertEqual(reward, 0)

        action = np.zeros((4,))
        if init_obs[2]:
            action[3] = 1
        else:
            action[2] = 1
        next_ob, reward, terminal, _ = env.step(action)
        self.assertEqual(reward, 0)

    def test_episode_length_is_right(self):
        env = OneCharMemory(n=3, num_steps=4, reward_for_remembering=10)
        action = env.reset()

        for _ in range(3):
            _, _, terminal, _ = env.step(action)
            self.assertFalse(terminal)

        _, _, terminal, _ = env.step(action)
        self.assertTrue(terminal)



if __name__ == '__main__':
    unittest.main()
