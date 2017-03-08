import tensorflow as tf

from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy


class _LstmLinearCell(tf.nn.rnn_cell.LSTMCell):
    """
    LSTM cell with a linear unit before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        super().__init__(num_units, **kwargs)
        self._output_dim = output_dim

    def __call__(self, *args, **kwargs):
        lstm_output, lstm_state = super().__call__(*args, **kwargs)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [self._num_units, self._output_dim])
            b = tf.get_variable('b', [self._output_dim],
                                initializer=tf.constant_initializer(0.0))

        env_action_logit = tf.matmul(lstm_output, W) + b
        return tf.nn.softmax(env_action_logit), lstm_state


class LstmMemoryPolicy(RnnCellPolicy):
    """
    write = affine function of environment observation and memory
    logits = affine function of environment observation, memory, and write
    action = softmax(logits)
    """

    def __init__(
            self,
            name_or_scope,
            action_dim,
            memory_dim,
            **kwargs
    ):
        assert memory_dim % 2 == 0
        self.setup_serialization(locals())
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self._num_lstm_units = self._memory_dim / 2
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        self._rnn_cell = _LstmLinearCell(
            self._num_lstm_units,
            self._action_dim,
        )
        lstm_memory_c_and_m_tuple = tf.split(1, 2, memory_obs)
        with tf.variable_scope("lstm") as self._rnn_cell_scope:
            cell_output = self._rnn_cell(env_obs, lstm_memory_c_and_m_tuple)
        env_action, (lstm_c_output, lstm_m_output) = cell_output
        write_action = tf.concat(1, (lstm_c_output, lstm_m_output))
        return env_action, write_action

    @property
    def rnn_cell(self):
        return self._rnn_cell

    def create_init_state_placeholder(self):
        lstm_init_c_state = tf.placeholder(
            tf.float32,
            [None, self._num_lstm_units],
            name='lstm_init_c_state',
        )
        lstm_init_m_state = tf.placeholder(
            tf.float32,
            [None, self._num_lstm_units],
            name='lstm_init_m_state',
        )
        return tf.concat(1, [lstm_init_c_state, lstm_init_m_state])


    @property
    def rnn_cell_scope(self):
        return self._rnn_cell_scope

