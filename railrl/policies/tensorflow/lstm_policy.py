import tensorflow as tf

from railrl.policies.tensorflow.nn_policy import NNPolicy


class LstmPolicy(NNPolicy):
    def __init__(
            self,
            name_or_scope,
            num_units,
            forget_bias=1.0,
            activation=tf.tanh,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self, observation_input):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(
            self._num_units,
            forget_bias=self._forget_bias,
            input_size=self.observation_dim,
            state_is_tuple=True,
            activation=self._activation
        )
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

        rnn_inputs = tf.unstack(tf.cast(self._x, tf.float32), axis=1)
        labels = tf.unstack(tf.cast(self._y, tf.float32), axis=1)

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

        self._total_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                                   labels=labels)
        self._train_step = tf.train.AdamOptimizer(
            self._learning_rate).minimize(
            self._total_loss)
