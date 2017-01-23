import unittest
import tensorflow as tf

from railrl.core.perceptron import Perceptron
from railrl.misc.tf_test_case import TFTestCase


class TestNeuralNetwork(TFTestCase):
    def var_names(self, vars):
        names = [var.name for var in vars]
        unique_names = set(names)
        self.assertEqual(
            len(names),
            len(unique_names),
            "Some variable names are repeated!"
        )
        return unique_names

    def test_get_params(self):
        in_size = 5
        out_size = 1
        W_name = "w"
        b_name = "b"
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            b_name=b_name,
        )

        all_vars = perceptron.get_params()
        names = self.var_names(all_vars)

        expected_names = {
            "perceptron/w:0",
            "perceptron/b:0",
        }
        self.assertEqual(names, expected_names)

    def test_regularize_only_w(self):
        in_size = 5
        out_size = 1
        W_name = "w"
        b_name = "b"
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            b_name=b_name,
        )

        all_vars = perceptron.get_params(regularizable=True)
        names = self.var_names(all_vars)

        expected_names = {
            "perceptron/w:0",
        }
        self.assertEqual(names, expected_names)

    def test_not_regularize_only_b(self):
        in_size = 5
        out_size = 1
        W_name = "w"
        b_name = "b"
        input = tf.placeholder(tf.float32, shape=(1, in_size))

        perceptron = Perceptron(
            "perceptron",
            input,
            in_size,
            out_size,
            W_name=W_name,
            b_name=b_name,
        )

        all_vars = perceptron.get_params(regularizable=False)
        names = self.var_names(all_vars)

        expected_names = {
            "perceptron/b:0",
        }
        self.assertEqual(names, expected_names)




if __name__ == '__main__':
    unittest.main()
