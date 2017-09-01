import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMCell, GRUCell

from railrl.core import tf_util
from railrl.core.rnn.rnn import LNLSTMCell
from railrl.core.rnn.rnn import RWACell
from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.policies.tensorflow.nn_policy import NNPolicy


class SplitterCell(RNNCell):
    """
    Remove some stuff from the end of the input before feeding it to the
    wrapped RNN.
    """
    def __init__(self, rnn_cell, num_or_size_splits):
        """
        :param rnn_cell: Some RNNCell
        :param num_or_size_splits: See
        https://www.tensorflow.org/api_docs/python/tf/split
        """
        self.rnn_cell = rnn_cell
        self.num_or_size_splits = num_or_size_splits

    def __call__(self, inputs, state, **kwargs):
        inputs = tf.split(
            inputs,
            self.num_or_size_splits,
            axis=1,
        )[0]
        return self.rnn_cell.__call__(inputs, state, **kwargs)

    @property
    def state_size(self):
        return self.rnn_cell.state_size

    @property
    def output_size(self):
        return self.rnn_cell.output_size


class LstmLinearCell(LNLSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            layer_norm=False,
            output_nonlinearity=tf.nn.tanh,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std
        self._layer_norm = layer_norm
        self.output_nonlinearity = output_nonlinearity

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            _, (lstm_output, lstm_state) = super().__call__(inputs, split_state,
                                                       scope=self.scope)

            if self._env_noise_std > 0.:
                lstm_output += self._env_noise_std * tf.random_normal(
                    tf.shape(lstm_output)
                )
            if self._memory_noise_std > 0.:
                lstm_state += self._memory_noise_std * tf.random_normal(
                    tf.shape(lstm_state)
                )

            flat_state = tf.concat(axis=1, values=(lstm_output, lstm_state))

            with tf.variable_scope('env_action') as self.env_action_scope:
                W = tf.get_variable('W', [self._num_units, self._output_dim])
                b = tf.get_variable('b', [self._output_dim],
                                    initializer=tf.constant_initializer(0.0))

            env_action_logit = tf.matmul(lstm_output, W) + b
            return self.output_nonlinearity(env_action_logit), flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class SeparateLstmLinearCell(LSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            output_nonlinearity=tf.nn.tanh,
            env_hidden_sizes=(100, 64),
            env_hidden_activation=tf.nn.relu,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std
        self.output_nonlinearity = output_nonlinearity
        self.write_action_scope = None
        self.env_action_scope = None
        self.env_hidden_sizes = env_hidden_sizes
        self.env_hidden_activation = env_hidden_activation

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            with tf.variable_scope('write_action') as self.write_action_scope:
                split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
                _, (lstm_output, lstm_state) = super().__call__(
                    inputs, split_state, scope=self.write_action_scope
                )

                if self._memory_noise_std > 0.:
                    lstm_state += self._memory_noise_std * tf.random_normal(
                        tf.shape(lstm_state)
                    )
                    lstm_output += self._memory_noise_std * tf.random_normal(
                        tf.shape(lstm_output)
                    )

                next_state = tf.concat(axis=1, values=(lstm_output, lstm_state))

            with tf.variable_scope('env_action') as self.env_action_scope:
                flat_inputs = tf.concat(axis=1, values=(inputs, state))
                env_output = tf_util.mlp(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    hidden_sizes=self.env_hidden_sizes,
                    nonlinearity=self.env_hidden_activation,
                    linear_output_size=self._output_dim
                )
                env_output = self.output_nonlinearity(env_output)
                if self._env_noise_std > 0.:
                    env_output += self._env_noise_std * tf.random_normal(
                        tf.shape(env_output)
                    )
            return env_output, next_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class FfResCell(RNNCell):
    """
    Feed forward + Residual connection.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            write_output_nonlinearity=tf.nn.tanh,
            write_hidden_sizes=(100, 64),
            write_hidden_activation=tf.nn.relu,
            env_output_nonlinearity=tf.nn.tanh,
            env_hidden_sizes=(100, 64),
            env_hidden_activation=tf.nn.relu,
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        self._state_dim = num_units
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std
        self.write_action_scope = None
        self.write_hidden_sizes = write_hidden_sizes
        self.write_hidden_activation = write_hidden_activation
        self.write_output_nonlinearity = write_output_nonlinearity
        self.env_action_scope = None
        self.env_hidden_sizes = env_hidden_sizes
        self.env_hidden_activation = env_hidden_activation
        self.env_output_nonlinearity = env_output_nonlinearity

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "ff_res_cell") as self.scope:
            flat_inputs = tf.concat(axis=1, values=(inputs, state))
            with tf.variable_scope('write_action') as self.write_action_scope:
                state_delta = tf_util.mlp(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    hidden_sizes=self.write_hidden_sizes,
                    nonlinearity=self.write_hidden_activation,
                    linear_output_size=self.state_size
                )
                state_delta = self.write_output_nonlinearity(state_delta)
                if self._memory_noise_std > 0.:
                    state_delta += self._memory_noise_std * tf.random_normal(
                        tf.shape(state_delta)
                    )
                next_state = state + state_delta

            with tf.variable_scope('env_action') as self.env_action_scope:
                env_output = tf_util.mlp(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    hidden_sizes=self.env_hidden_sizes,
                    nonlinearity=self.env_hidden_activation,
                    linear_output_size=self._output_dim
                )
                env_output = self.env_output_nonlinearity(env_output)
                if self._env_noise_std > 0.:
                    env_output += self._env_noise_std * tf.random_normal(
                        tf.shape(env_output)
                    )
            return env_output, next_state

    @property
    def state_size(self):
        return self._state_dim

    @property
    def output_size(self):
        return self._output_dim


class LstmMlpCell(LSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            _, (lstm_output, lstm_state) = super().__call__(inputs, split_state,
                                                            scope=self.scope)

            if self._env_noise_std > 0.:
                lstm_output += self._env_noise_std * tf.random_normal(
                    tf.shape(lstm_output)
                )
            if self._memory_noise_std > 0.:
                lstm_state += self._memory_noise_std * tf.random_normal(
                    tf.shape(lstm_state)
                )

            flat_state = tf.concat(axis=1, values=(lstm_output, lstm_state))

            with tf.variable_scope('env_action') as self.env_action_scope:
                env_action_logit = tf_util.mlp(
                    lstm_output,
                    self._num_units,
                    [100, 64, self._output_dim],
                    tf.nn.relu,
                )
            return tf.nn.softmax(env_action_logit), flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class LstmLinearCellNoiseAll(LSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            raw_lstm_output, (state, lstm_output) = super().__call__(
                inputs, split_state, scope=self.scope
            )

            if self._memory_noise_std > 0.:
                state += self._memory_noise_std * tf.random_normal(
                    tf.shape(state)
                )
                lstm_output += self._memory_noise_std * tf.random_normal(
                    tf.shape(lstm_output)
                )
            flat_state = tf.concat(axis=1, values=(state, lstm_output))

            # Apparently TensorFlow doesn't support the following:
            # TODO(vitchyr): open a bug report. I guess they can't do the
            # reparameterization trick through concat
            # flat_state = tf.concat(axis=1, values=(state, lstm_output))
            # if self._memory_noise_std > 0.:
            #     flat_state += self._memory_noise_std * tf.random_normal(
            #         tf.shape(flat_state)
            #     )

            with tf.variable_scope('env_action') as self.env_action_scope:
                W = tf.get_variable('W', [self._num_units, self._output_dim])
                b = tf.get_variable('b', [self._output_dim],
                                    initializer=tf.constant_initializer(0.0))

            if self._env_noise_std > 0.:
                raw_lstm_output += self._env_noise_std * tf.random_normal(
                    tf.shape(raw_lstm_output)
                )
            env_action_logit = tf.matmul(raw_lstm_output, W) + b

            return tf.nn.softmax(env_action_logit), flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class LstmLinearCellNoiseAllNoiseLogit(LSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            lstm_output, full_state = super().__call__(inputs, split_state,
                                                       scope=self.scope)

            flat_state = tf.concat(axis=1, values=full_state)
            if self._memory_noise_std > 0.:
                flat_state += self._memory_noise_std * tf.random_normal(
                    tf.shape(flat_state)
                )

            with tf.variable_scope('env_action') as self.env_action_scope:
                W = tf.get_variable('W', [self._num_units, self._output_dim])
                b = tf.get_variable('b', [self._output_dim],
                                    initializer=tf.constant_initializer(0.0))

            env_action_logit = tf.matmul(lstm_output, W) + b
            if self._env_noise_std > 0.:
                env_action_logit += self._env_noise_std * tf.random_normal(
                    tf.shape(env_action_logit)
                )

            return tf.nn.softmax(env_action_logit), flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class LstmLinearCellSwapped(LSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            _, (lstm_state, lstm_output) = super().__call__(inputs, split_state,
                                                            scope=self.scope)

            if self._env_noise_std > 0.:
                lstm_output += self._env_noise_std * tf.random_normal(
                    tf.shape(lstm_output)
                )
            if self._memory_noise_std > 0.:
                lstm_state += self._memory_noise_std * tf.random_normal(
                    tf.shape(lstm_state)
                )

            flat_state = tf.concat(axis=1, values=(lstm_state, lstm_output))

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


class GRULinearCell(GRUCell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            output, new_state = super().__call__(inputs, state,
                                                 scope=self.scope)

            if self._memory_noise_std > 0.:
                new_state += self._memory_noise_std * tf.random_normal(
                    tf.shape(new_state)
                )

            with tf.variable_scope('env_action') as self.env_action_scope:
                W = tf.get_variable('W', [self._num_units, self._output_dim])
                b = tf.get_variable('b', [self._output_dim],
                                    initializer=tf.constant_initializer(0.0))

            env_action_logit = tf.matmul(output, W) + b
            if self._env_noise_std > 0.:
                output += self._env_noise_std * tf.random_normal(
                    tf.shape(output)
                )

            return tf.nn.softmax(env_action_logit), new_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_dim

class ResidualLstmLinearCell(LSTMCell):
    """
    LSTM cell with a linear unit + softmax before the output. However,
    the state is additive rather than just copied over.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        super().__init__(num_units / 2, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            _, (lstm_output, lstm_state) = super().__call__(inputs, split_state,
                                                            scope=self.scope)

            if self._env_noise_std > 0.:
                lstm_output += self._env_noise_std * tf.random_normal(
                    tf.shape(lstm_output)
                )
            if self._memory_noise_std > 0.:
                lstm_state += self._memory_noise_std * tf.random_normal(
                    tf.shape(lstm_state)
                )

            flat_state = tf.concat(
                axis=1,
                values=(lstm_output, lstm_state + split_state[1])
            )

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


class OutputAwareLstmCell(LSTMCell):
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


class FrozenHiddenLstmLinearCell(LSTMCell):
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
        self._num_units = int(num_units)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_rnn_cell"):
            flat_inputs = tf.concat(axis=1, values=[inputs, state])
            with tf.variable_scope('env_action') as self.env_action_scope:
                env_action_logit = tf_util.linear(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    self.output_size,
                )
            with tf.variable_scope('next_state'):
                next_state = tf_util.linear(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    self.state_size,
                )

        return tf.nn.softmax(env_action_logit), next_state

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_dim


class DebugCell(tf.contrib.rnn.RNNCell):
    """
    Used for debugging.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            **kwargs
    ):
        self._output_dim = int(output_dim)
        self._num_units = int(num_units)

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "debug_cell"):
            with tf.variable_scope('env_action') as self.env_action_scope:
                env_action_var = tf_util.weight_variable(
                    [inputs.get_shape()[-1], self.output_size],
                    initializer=tf.constant_initializer(1.),
                    name='env_out',
                )
                env_output = tf.matmul(inputs, env_action_var) + 1
            with tf.variable_scope('next_state') as self.write_action_scope:
                write_var = tf_util.weight_variable(
                    [state.get_shape()[-1], self.state_size],
                    initializer=tf.constant_initializer(0.),
                    name='write',
                )
                write = state + tf.matmul(state, write_var) + 1

        return env_output, write

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_dim


class SeparateRWALinearCell(RWACell):
    """
    LSTM cell with a linear unit + softmax before the output.
    """
    def __init__(
            self,
            num_units,
            output_dim,
            env_noise_std=0.,
            memory_noise_std=0.,
            output_nonlinearity=tf.nn.tanh,
            env_hidden_sizes=(100, 64),
            env_hidden_activation=tf.nn.relu,
            state_is_flat_externally=True,
            **kwargs
    ):
        assert env_noise_std >= 0.
        assert memory_noise_std >= 0.
        if state_is_flat_externally:
            assert num_units % 4 == 0
            super().__init__(num_units // 4, **kwargs)
        else:
            super().__init__(num_units, **kwargs)
        self._output_dim = output_dim
        self._env_noise_std = env_noise_std
        self._memory_noise_std = memory_noise_std
        self.output_nonlinearity = output_nonlinearity
        self.write_action_scope = None
        self.env_action_scope = None
        self.env_hidden_sizes = env_hidden_sizes
        self.env_hidden_activation = env_hidden_activation
        self.state_is_flat_externally = state_is_flat_externally

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "linear_lstm") as self.scope:
            with tf.variable_scope('write_action') as self.write_action_scope:
                if self.state_is_flat_externally:
                    state = tf.split(axis=1, num_or_size_splits=4, value=state)
                _, next_state = super().__call__(
                    inputs, state
                )

                if self._memory_noise_std > 0.:
                    next_state = tuple(
                        state_comp + self._memory_noise_std * tf.random_normal(
                            tf.shape(state_comp)
                        )
                        for state_comp in next_state
                    )

                if self.state_is_flat_externally:
                    next_state = tf.concat(axis=1, values=next_state)

            with tf.variable_scope('env_action') as self.env_action_scope:
                all_inputs = [inputs] + list(state)
                flat_inputs = tf.concat(axis=1, values=all_inputs)
                env_output = tf_util.mlp(
                    flat_inputs,
                    flat_inputs.get_shape()[-1],
                    hidden_sizes=self.env_hidden_sizes,
                    nonlinearity=self.env_hidden_activation,
                    linear_output_size=self._output_dim
                )
                env_output = self.output_nonlinearity(env_output)
                if self._env_noise_std > 0.:
                    env_output += self._env_noise_std * tf.random_normal(
                        tf.shape(env_output)
                    )
            return env_output, next_state

    @property
    def state_size(self):
        if self.state_is_flat_externally:
            return 4 * super().state_size[0]
        return super().state_size

    @property
    def output_size(self):
        return self._output_dim

class LstmMemoryPolicy(RnnCellPolicy):
    """
    write = LSTM function of environment observation and memory
    logits = function of environment observation and memory
    action = softmax(logits)
    """

    def __init__(
            self,
            name_or_scope,
            action_dim,
            memory_dim,
            num_env_obs_dims_to_use,
            init_state=None,
            rnn_cell_class=LstmLinearCell,
            rnn_cell_params=None,
            **kwargs
    ):
        if rnn_cell_params is None:
            rnn_cell_params = {}
        assert memory_dim % 2 == 0
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self._num_env_obs_dims_to_use = num_env_obs_dims_to_use
        self._num_env_obs_dims_to_ignore = (
            self.observation_dim[0] - num_env_obs_dims_to_use
        )
        assert self._num_env_obs_dims_to_ignore >= 0
        self.rnn_cell_class = rnn_cell_class
        self.rnn_cell_params = rnn_cell_params
        self.init_state = self._placeholder_if_none(
            init_state,
            [None, self._memory_dim],
            name='lstm_init_state',
            dtype=tf.float32,
        )
        # self._env_action_scope_name = None
        # self._write_action_scope_name = None
        self._create_network()

    def _create_network_internal(self, observation_input=None, init_state=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        self._rnn_cell = self.rnn_cell_class(
            self._memory_dim,
            self._action_dim,
            **self.rnn_cell_params
        )
        if self._num_env_obs_dims_to_ignore > 0:
            self._rnn_cell = SplitterCell(
                self._rnn_cell,
                [
                    self._num_env_obs_dims_to_use,
                    self._num_env_obs_dims_to_ignore,
                ],
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
        if self._num_env_obs_dims_to_ignore > 0:
            self._env_action_scope_name = (
                self._rnn_cell.rnn_cell.env_action_scope.name
            )
            self._write_action_scope_name = (
                self._rnn_cell.rnn_cell.write_action_scope.name
            )
        else:
            self._env_action_scope_name = self._rnn_cell.env_action_scope.name
            self._write_action_scope_name = (
                self._rnn_cell.write_action_scope.name
            )
        return cell_output

    def get_params_internal(self, env_only=False, write_only=False):
        assert not (env_only and write_only)
        if env_only:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self._env_action_scope_name)
        if write_only:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     self._write_action_scope_name)
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


class FlatLstmMemoryPolicy(NNPolicy):
    def __init__(
            self,
            name_or_scope,
            action_dim,
            memory_dim,
            env_obs_dim,
            init_state=None,
            rnn_cell_class=LstmLinearCell,
            rnn_cell_params=None,
            **kwargs
    ):
        if rnn_cell_params is None:
            rnn_cell_params = {}
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._env_obs_dim = env_obs_dim
        self.rnn_cell_class = rnn_cell_class
        self.rnn_cell_params = rnn_cell_params
        self.init_state = self._placeholder_if_none(
            init_state,
            [None, self._memory_dim],
            name='lstm_init_state',
            dtype=tf.float32,
        )
        self._create_network()

    def _create_network_internal(self, observation_input=None):
        env_obs, memory_obs = tf.split(
            observation_input,
            [self._env_obs_dim, self._memory_dim],
            axis=1,
        )
        self._rnn_cell = self.rnn_cell_class(
            self._memory_dim,
            self._action_dim,
            **self.rnn_cell_params
        )
        with tf.variable_scope("rnn_cell") as self._rnn_cell_scope:
            cell_output = self._rnn_cell(env_obs, memory_obs)
        return tf.concat(cell_output, axis=1)
