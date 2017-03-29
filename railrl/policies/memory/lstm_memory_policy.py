import tensorflow as tf

from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy


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
        assert memory_dim == action_dim * 2
        self.setup_serialization(locals())
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self._num_lstm_units = self._action_dim
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        tf.contrib.rnn
        self._rnn_cell = tf.contrib.rnn.BasicLSTMCell(
            self._num_lstm_units,
            state_is_tuple=True,
        )
        lstm_memory_c_and_m_tuple = tf.split(axis=1, num_or_size_splits=2, value=memory_obs)
        with tf.variable_scope("lstm") as self._rnn_cell_scope:
            cell_output = self._rnn_cell(env_obs, lstm_memory_c_and_m_tuple)
        env_action_logits, (lstm_c_output, lstm_m_output) = cell_output
        write_action = tf.concat(axis=1, values=(lstm_c_output, lstm_m_output))
        env_action = tf.nn.softmax(env_action_logits)
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
        return tf.concat(axis=1, values=[lstm_init_c_state, lstm_init_m_state])


    @property
    def rnn_cell_scope(self):
        return self._rnn_cell_scope

