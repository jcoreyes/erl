from collections import OrderedDict
import numpy as np
import tensorflow as tf
import time

from railrl.envs.supervised_learning_env import SupervisedLearningEnv
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger
from rllab.misc.overrides import overrides


class Bptt(RLAlgorithm):
    """
    Back propagation through time
    """

    def __init__(
            self,
            env: SupervisedLearningEnv,
            num_batches_per_epoch=32,
            num_epochs=1000,
            learning_rate=1e-3,
            batch_size=32,
            eval_num_batches=4,
            lstm_state_size=10,
    ):
        self._num_batches_per_epoch = num_batches_per_epoch
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._eval_num_batches = eval_num_batches
        self._state_size = lstm_state_size
        self._env = env
        self._num_classes = env.feature_dim
        self._num_steps = env.sequence_length

        self._training_losses = []

    @overrides
    def train(self):
        start_time = time.time()
        self._init_ops()
        logger.log("Graph creation time: {0}".format(time.time() - start_time))
        for epoch in range(self._num_epochs):
            logger.push_prefix('Epoch #%d | ' % epoch)

            start_time = time.time()
            for update in range(self._num_batches_per_epoch):
                X, Y = self._env.get_batch(self._batch_size)
                self._bptt_train(X, Y)
            logger.log("Training time: {0}".format(time.time() - start_time))

            start_time = time.time()
            self._eval(epoch)
            logger.pop_prefix()
            logger.log("Eval time: {0}".format(time.time() - start_time))

    def _init_ops(self):
        self._sess = tf.get_default_session() or tf.Session()
        self._init_network()
        self._sess.run(tf.global_variables_initializer())

    def _init_network(self):
        """
        Implementation based on
        http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
        """

        self._x = tf.placeholder(
            tf.int32,
            [self._batch_size, self._num_steps, self._num_classes],
            name='input_placeholder',
        )
        self._y = tf.placeholder(
            tf.int32,
            [self._batch_size, self._num_steps, self._num_classes],
            name='labels_placeholder',
        )

        rnn_inputs = tf.unpack(tf.cast(self._x, tf.float32), axis=1)
        labels = tf.unpack(tf.cast(self._y, tf.float32), axis=1)

        cell = tf.nn.rnn_cell.LSTMCell(self._state_size, state_is_tuple=True)
        init_state = cell.zero_state(self._batch_size, tf.float32)
        rnn_outputs, self._final_state = tf.nn.rnn(
            cell,
            rnn_inputs,
            initial_state=init_state,
        )
        self._c_init_state, self._m_init_state = init_state

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self._state_size, self._num_classes])
            b = tf.get_variable('b', [self._num_classes],
                                initializer=tf.constant_initializer(0.0))
        logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        self._predictions = [tf.nn.softmax(logit) for logit in logits]

        self._total_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits,
                                                                   labels)
        self._train_step = tf.train.AdamOptimizer(
            self._learning_rate).minimize(
            self._total_loss)

    def _bptt_train(self, X, Y):
        c_init_state = np.zeros((self._batch_size, self._state_size))
        m_init_state = np.zeros((self._batch_size, self._state_size))
        training_loss_, _ = self._sess.run(
            [
                self._total_loss,
                self._train_step,
            ],
            feed_dict={
                self._x: X,
                self._y: Y,
                self._c_init_state: c_init_state,
                self._m_init_state: m_init_state,
            },
        )
        self._training_losses.append(training_loss_)

    def _eval(self, epoch):

        # Create a tuple for c_state and m_state
        def sample_eval_loss():
            X, Y = self._env.get_batch(self._batch_size)
            c_init_state = np.zeros((self._batch_size, self._state_size))
            m_init_state = np.zeros((self._batch_size, self._state_size))
            return self._sess.run(
                self._total_loss,
                feed_dict={
                    self._x: X,
                    self._y: Y,
                    self._c_init_state: c_init_state,
                    self._m_init_state: m_init_state,
                },
            )

        eval_loss = np.mean(
            [sample_eval_loss() for _ in range(self._eval_num_batches)]
        )
        last_statistics = OrderedDict([
            ('Epoch', epoch),
            ('Training Loss', np.mean(self._training_losses)),
            ('Eval Loss', eval_loss),
        ])
        self._training_losses = []

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)
        logger.dump_tabular(with_prefix=False)
