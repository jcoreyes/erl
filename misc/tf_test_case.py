import unittest
import tensorflow as tf

from misc.testing_utils import are_np_arrays_equal


class TFTestCase(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.sess = tf.get_default_session() or tf.Session()
        self.sess_context = self.sess.as_default()
        self.sess_context.__enter__()

    def tearDown(self):
        self.sess_context.__exit__(None, None, None)
        self.sess.close()

    def assertNpEqual(self, np_arr1, np_arr2):
        self.assertTrue(
            are_np_arrays_equal(np_arr1, np_arr2),
            "Numpy arrays not equal")
