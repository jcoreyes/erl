import tensorflow as tf
import numpy as np

from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.core import tf_util


class LstmLinearCell(tf.contrib.rnn.BasicLSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            lstm_output, lstm_state = super().__call__(inputs, split_state,
                                                       scope=self.scope)
            flat_state = tf.concat(axis=1, values=lstm_state)

            with tf.variable_scope('env_action') as self.env_action_scope:
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


class OutputAwareLstmCell(tf.contrib.rnn.BasicLSTMCell):
    """
    Env action = linear function of input.
    LSTM input = env action and env observation.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            with tf.variable_scope('env_action') as self.env_action_scope:
                flat_inputs = tf.concat(axis=1, values=[inputs, state])
                env_action_logit = tf_util.linear(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    self._output_dim,
                )

            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            lstm_input = tf.concat(values=(inputs, env_action_logit), axis=1)
            lstm_output, lstm_state = super().__call__(lstm_input,
                                                       split_state,
                                                       scope=self.scope)
            flat_state = tf.concat(axis=1, values=lstm_state)

            return tf.nn.softmax(env_action_logit), flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class FrozenHiddenLstmLinearCell(tf.contrib.rnn.BasicLSTMCell):
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "frozen_hidden_lstm_lin") as new_scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            lstm_output, lstm_state = super().__call__(inputs, split_state,
                                                       scope=new_scope)
            flat_state = tf.concat(axis=1, values=lstm_state)

            all_inputs = tf.concat(axis=1, values=(inputs, state))
            with tf.variable_scope('env_action') as self.env_action_scope:
                env_action_logit = tf_util.mlp(
                    all_inputs,
                    all_inputs.get_shape()[-1],
                    (32, 32, self._output_dim),
                    tf.nn.tanh,
                )
        return tf.nn.softmax(env_action_logit), flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class IRnnCell(tf.contrib.rnn.RNNCell):
    def __init__(
            self,
            num_units,
            output_dim,
            num_hidden_layers=2,
    ):
        self._output_dim = int(output_dim)
        self.num_units = int(num_units)
        self.num_hidden_layers = num_hidden_layers

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "irnn_cell"):
            with tf.variable_scope('env_action') as self.env_action_scope:
                flat_inputs = tf.concat(axis=1, values=[inputs, state])
                env_action_logit = tf_util.linear(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    self._output_dim,
                )

            """
            Set up IRNN inputs and weights
            """
            irnn_inputs = tf.concat(values=(inputs, env_action_logit), axis=1)
            W_hidden = tf_util.weight_variable(
                [self.num_units, self.num_units],
                initializer=tf.constant_initializer(
                    value=np.eye(self.num_units),
                    dtype=tf.float32,
                ),
                name="W_hidden",
            )
            b_state = tf_util.bias_variable(
                self.num_units,
                initializer=tf.constant_initializer(0.),
                name="b_state",
            )
            W_input = tf_util.weight_variable(
                [irnn_inputs.get_shape()[-1], self.num_units],
                name="W_input",
                initializer=tf_util.xavier_uniform_initializer()
            )

            """
            Compute the next state
            """
            last_layer = (
                tf.matmul(state, W_hidden)
                + tf.matmul(irnn_inputs, W_input)
                + b_state
            )
            for _ in range(self.num_hidden_layers):
                last_layer = tf.nn.relu(last_layer)
                last_layer = tf.matmul(last_layer, W_hidden) + b_state
            next_state = last_layer

        return tf.nn.softmax(env_action_logit), next_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_dim


class LinearRnnCell(tf.contrib.rnn.RNNCell):
    def __init__(
            self,
            num_units,
            output_dim,
    ):
        self._output_dim = int(output_dim)
        self.num_units = int(num_units)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_rnn_cell"):
            flat_inputs = tf.concat(axis=1, values=[inputs, state])
            with tf.variable_scope('env_action') as self.env_action_scope:
                env_action_logit = tf_util.linear(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    self._output_dim,
                )
            with tf.variable_scope('next_state'):
                next_state = tf_util.linear(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    self._output_dim,
                )

        return tf.nn.softmax(env_action_logit), next_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_dim


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
            rnn_cell_class=LstmLinearCell,
            **kwargs
    ):
        assert memory_dim % 2 == 0
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self.rnn_cell_class = rnn_cell_class
        self.init_state = self._placeholder_if_none(
            init_state,
            [None, self._memory_dim],
            name='lstm_init_state',
            dtype=tf.float32,
        )
        self._create_network()

    def _create_network_internal(self, observation_input=None, init_state=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        self._rnn_cell = self.rnn_cell_class(
            self._memory_dim,
            self._action_dim,
        )
        # TODO(vitchyr): I'm pretty sure that this rnn_cell_scope should NOT
        # be passed into the _rnn_cell method. Basically, it should be passed
        # to other functions like static_rnn. It seems that the scope that
        # should be passed into static_rnn is the scope *surrounding* the
        # call to the rnn's __call__ method, and NOT the scope that should be
        # passed in to the __call__ method.
        #
        # I should verify this.
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
