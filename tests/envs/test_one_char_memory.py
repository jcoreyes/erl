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

    def test_middle_of_batches_are_all_zeros(self):
        env = OneCharMemory(n=3, num_steps=4)
        X, Y = env.get_batch(batch_size=100)

        self.assertNpArrayConstant(X[:, 1:, :], 0)
        self.assertNpArrayConstant(Y[:, :-1, :], 0)

    def test_first_x_is_one_hot(self):
        env = OneCharMemory(n=3, num_steps=4)
        X, Y = env.get_batch(batch_size=100)

        self.assertNpArrayConstant(np.sum(np.array(1 == X[:, 0, :]), axis=1), 1)



if __name__ == '__main__':
    unittest.main()
