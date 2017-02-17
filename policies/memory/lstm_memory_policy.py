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
            memory_dim,
            action_dim,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self._num_lstm_units = 5
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        self._rnn_cell = tf.nn.rnn_cell.LSTMCell(
            self._num_lstm_units,
            state_is_tuple=True,
        )
        with tf.variable_scope("lstm") as self._rnn_cell_scope:
            cell_output = self._rnn_cell(env_obs, memory_obs)
        return cell_output

    @property
    def rnn_cell(self):
        return self._rnn_cell

    @property
    def rnn_cell_scope(self):
        return self._rnn_cell_scope

