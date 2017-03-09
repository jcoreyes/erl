"""
Show an example of how to create a TensorBoard graph.
"""
from railrl.launchers.launcher_util import run_experiment
import tensorflow as tf
from railrl.core.tf_util import linear
from rllab.misc import logger


def example(*_):
    def create_network(last_layer):
        return linear(last_layer, 5, 5)
    sess = tf.Session()
    input_a = tf.placeholder(tf.float32, [None, 5], 'input_a')
    input_b = tf.placeholder(tf.float32, [None, 5], 'input_b')
    with tf.variable_scope('a'):
        x = create_network(input_a)
    # Note: whenever you reuse a scope, a second node will appear with "_1"
    # appended to it. That just means that the two are tied.
    with tf.variable_scope('a', reuse=True):
        y = create_network(input_b)
    tf.summary.FileWriter(logger.get_snapshot_dir(), sess.graph)


if __name__ == "__main__":
    run_experiment(
        example,
        exp_prefix="generate-tensorboard-graph",
        mode='here',
    )
