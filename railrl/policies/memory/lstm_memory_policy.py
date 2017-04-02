import tensorflow as tf

from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.core.tf_util import mlp


class _LstmLinearCell(tf.contrib.rnn.BasicLSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        super().__init__(num_units, **kwargs)
        self._output_dim = output_dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            lstm_output, lstm_state = super().__call__(inputs, split_state,
                                                       scope=self.scope)
            flat_state = tf.concat(axis=1, values=lstm_state)

            with tf.variable_scope('softmax'):
                W = tf.get_variable('W', [self._num_units, self._output_dim])
                b = tf.get_variable('b', [self._output_dim],
                                    initializer=tf.constant_initializer(0.0))

            env_action_logit = tf.matmul(lstm_output, W) + b
            return tf.nn.softmax(env_action_logit), flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class _FrozenHiddenLstmLinearCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        super().__init__(num_units, **kwargs)
        self._output_dim = output_dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "frozen_hidden_lstm_lin") as new_scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            lstm_output, lstm_state = super().__call__(inputs, split_state,
                                                       scope=new_scope)
            flat_state = tf.concat(axis=1, values=lstm_state)

            all_inputs = tf.concat(axis=1, values=(inputs, state))
            with tf.variable_scope('env_action') as self.env_action_scope:
                env_action_logit = mlp(
                    all_inputs,
                    all_inputs.get_shape()[-1],
                    (32, 32, self._output_dim),
                    tf.nn.tanh,
                )
        return tf.nn.softmax(env_action_logit), flat_state


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
            init_state=None,
            freeze_hidden=False,
            **kwargs
    ):
        assert memory_dim % 2 == 0
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self._num_lstm_units = self._memory_dim / 2
        self.freeze_hidden = freeze_hidden
        self.init_state = self._placeholder_if_none(
            init_state,
            [None, self._num_lstm_units * 2],
            name='lstm_init_state',
            dtype=tf.float32,
        )
        self._create_network()

    def _create_network_internal(self, observation_input=None, init_state=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        if self.freeze_hidden:
            self._rnn_cell = _FrozenHiddenLstmLinearCell(
                self._num_lstm_units,
                self._action_dim,
            )
        else:
            self._rnn_cell = _LstmLinearCell(
                self._num_lstm_units,
                self._action_dim,
            )
        with tf.variable_scope("rnn_cell") as self._rnn_cell_scope:
            cell_output = self._rnn_cell(env_obs, memory_obs)
        return cell_output

    def get_params_internal(self, env_only=False):
        if env_only:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self._rnn_cell.env_action_scope.name)
        else:
            return super().get_params_internal()

    @property
    def rnn_cell(self):
        return self._rnn_cell

    def get_init_state_placeholder(self):
        return self.init_state

    @property
    def rnn_cell_scope(self):
        return self._rnn_cell_scope

    @property
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
            init_state=self.init_state,
        )
