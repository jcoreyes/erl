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
            eval_batch_size=128,
            lstm_state_size=10,
    ):
        self._num_batches_per_epoch = num_batches_per_epoch
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._eval_batch_size = eval_batch_size
        self._state_size = lstm_state_size
        self._env = env
        self._num_classes = env.feature_dim
        self._num_steps = env.sequence_length

        self._training_losses = []

    @overrides
    def train(self):
        self._init_ops()
        for epoch in range(self._num_epochs):
            logger.push_prefix('Epoch #%d | ' % epoch)

            start_time = time.time()
            for update in range(self._num_batches_per_epoch):
                X, Y = self._env.get_batch(self._batch_size)
                self._bptt_train(X, Y)
            logger.log("Training time: {0}".format(time.time() - start_time))

            start_time = time.time()
            X, Y = self._env.get_batch(self._eval_batch_size)
            self._eval(X, Y, epoch)
            logger.pop_prefix()
            logger.log("Eval time: {0}".format(time.time() - start_time))

    def _init_ops(self):
        self._sess = tf.get_default_session() or tf.Session()
        self._init_network()

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
        self._init_state = tf.zeros([self._batch_size, self._state_size])

        """
        Inputs/outputs
        """

        rnn_inputs = tf.unpack(self._x, axis=1)
        rnn_targets = tf.unpack(self._y, axis=1)

        """
        RNN
        """

        cell = tf.nn.rnn_cell.LSTMCell(self._state_size)
        rnn_outputs, self._final_state = tf.nn.rnn(
            cell,
            rnn_inputs,
            initial_state=self._init_state,
        )

        """
        Predictions, loss, training step
        """

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self._state_size, self._num_classes])
            b = tf.get_variable('b', [self._num_classes],
                                initializer=tf.constant_initializer(0.0))
        logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        self._predictions = [tf.nn.softmax(logit) for logit in logits]

        # weight all losses equally
        loss_weights = [tf.ones([self._batch_size]) for _ in
                        range(self._num_steps)]
        losses = tf.nn.seq2seq.sequence_loss_by_example(logits,
                                                        rnn_targets,
                                                        loss_weights)
        self._total_loss = tf.reduce_mean(losses)
        self._train_step = tf.train.AdamOptimizer(
            self._learning_rate).minimize(
            self._total_loss)

    def _bptt_train(self, X, Y):
        init_state = np.zeros((self._batch_size, self._state_size))
        training_loss_, _ = self._sess.run(
            [
                self._total_loss,
                self._train_step,
            ],
            feed_dict={
                self._x: X,
                self._y: Y,
                self._init_state: init_state,
            },
        )
        self._training_losses.append(training_loss_)

    def _eval(self, X, Y, epoch):
        init_state = np.zeros((self._batch_size, self._state_size))
        eval_loss, _ = self._sess.run(
            self._total_loss,
            feed_dict={
                self._x: X,
                self._y: Y,
                self._init_state: init_state,
            },
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
