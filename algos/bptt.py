from collections import OrderedDict
import numpy as np
import tensorflow as tf
import time

from railrl.misc.data_processing import create_stats_ordered_dict
from rllab.algos.base import RLAlgorithm
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.core.parameterized import Parameterized

__BPTT_VARIABLE_SCOPE__ = "bptt_variable_scope"


class Bptt(Parameterized, RLAlgorithm, Serializable):
    """
    Back propagation through time
    """

    def __init__(
            self,
            env,
            num_batches_per_epoch=32,
            num_epochs=10000,
            learning_rate=1e-3,
            batch_size=32,
            eval_num_episodes=64,
            lstm_state_size=10,
            **kwargs
    ):
        super().__init__(**kwargs)
        Serializable.quick_init(self, locals())
        self._num_batches_per_epoch = num_batches_per_epoch
        self._num_epochs = num_epochs
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._eval_num_episodes = eval_num_episodes
        self._state_size = lstm_state_size
        self._env = env
        self._num_classes = env.feature_dim
        self._num_steps = env.sequence_length

        self._training_losses = []
        self._sess = None

        start_time = time.time()
        self._init_ops()
        logger.log("Graph creation time: {0}".format(time.time() - start_time))

    @overrides
    def train(self):
        with self._sess.as_default():
            for epoch in range(self._num_epochs):
                logger.push_prefix('Epoch #%d | ' % epoch)

                start_time = time.time()
                for update in range(self._num_batches_per_epoch):
                    X, Y = self._env.get_batch(self._batch_size)
                    self._bptt_train(X, Y)
                logger.log(
                    "Training time: {0}".format(time.time() - start_time))

                start_time = time.time()
                self._eval(epoch)
                logger.pop_prefix()
                logger.log("Eval time: {0}".format(time.time() - start_time))

                params = self.get_epoch_snapshot(epoch)
                logger.save_itr_params(epoch, params)

    def _init_ops(self):
        self._sess = tf.get_default_session() or tf.Session()
        with self._sess.as_default():
            with tf.variable_scope(__BPTT_VARIABLE_SCOPE__):
                self._init_network()
        self._sess.run(tf.global_variables_initializer())

    def _init_network(self):
        """
        Implementation based on
        http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
        """

        self._x = tf.placeholder(
            tf.int32,
            [None, self._num_steps, self._num_classes],
            name='input_placeholder',
        )
        self._y = tf.placeholder(
            tf.int32,
            [None, self._num_steps, self._num_classes],
            name='labels_placeholder',
        )

        rnn_inputs = tf.unpack(tf.cast(self._x, tf.float32), axis=1)
        labels = tf.unpack(tf.cast(self._y, tf.float32), axis=1)

        cell = tf.nn.rnn_cell.LSTMCell(self._state_size, state_is_tuple=True)
        rnn_outputs, self._final_state = tf.nn.rnn(
            cell,
            rnn_inputs,
            dtype=tf.float32,
        )

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
        training_loss_, _ = self._sess.run(
            [
                self._total_loss,
                self._train_step,
            ],
            feed_dict={
                self._x: X,
                self._y: Y,
            },
        )
        self._training_losses.append(training_loss_)

    def _eval(self, epoch):
        X, Y = self._env.get_batch(self._eval_num_episodes)
        eval_losses, predictions = self._sess.run(
            [
                self._total_loss,
                self._predictions,
            ],
            feed_dict={
                self._x: X,
                self._y: Y,
            },
        )
        target_onehots = Y[:, -1, :]
        final_predictions = predictions[-1]  # batch_size X dim
        nonfinal_predictions = predictions[:-1]  # list of batch_size X dim
        nonfinal_predictions_sequence_dimension_flattened = np.vstack(
            nonfinal_predictions
        )  # shape = N X dim
        nonfinal_prob_zero = [softmax[0] for softmax in
                              nonfinal_predictions_sequence_dimension_flattened]

        """
        Short version would be the following, but I don't think it's very
        readable:
        ```
        # Axis 0 is batch size, so take the argmax over axis 1
        final_probs_correct = final_predictions[0, np.argmax(target_onehots,
                                                             axis=1)]
        ```
        """
        final_probs_correct = []
        for final_prediction, target_onehot in zip(final_predictions,
                                                   target_onehots):
            correct_pred_idx = np.argmax(target_onehot)
            final_probs_correct.append(final_prediction[correct_pred_idx])
        final_prob_zero = [softmax[0] for softmax in final_predictions]

        last_statistics = OrderedDict([
            ('Epoch', epoch),
        ])
        last_statistics.update(create_stats_ordered_dict('Training Loss',
                                                         self._training_losses))
        last_statistics.update(create_stats_ordered_dict('Eval Loss',
                                                         eval_losses))
        last_statistics.update(create_stats_ordered_dict(
            'Final P(correct)',
            final_probs_correct))
        last_statistics.update(create_stats_ordered_dict(
            'Non-final P(zero)',
            nonfinal_prob_zero))
        last_statistics.update(create_stats_ordered_dict(
            'Final P(zero)',
            final_prob_zero))
        self._training_losses = []

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)
        logger.dump_tabular(with_prefix=False)

    def get_epoch_snapshot(self, epoch):
        return dict(
            epoch=epoch,
            bptt=self,
        )

    def get_prediction(self, X):
        return self._sess.run(self._predictions,
                              feed_dict={
                                  self._x: X,
                              })

    def get_params_internal(self, **tags):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 __BPTT_VARIABLE_SCOPE__)
