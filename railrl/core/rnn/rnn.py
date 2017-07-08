import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.contrib.layers import layer_norm
from railrl.core.tf_util import layer_normalize


class SaveOutputRnn(tf.contrib.rnn.RNNCell):
    """
    An RNN that wraps another RNN. This RNN saves the last output in the
    state (in addition to the normal state).
    """

    def __init__(
            self,
            rnn_cell: tf.contrib.rnn.RNNCell,
    ):
        self._wrapped_rnn_cell = rnn_cell

    def __call__(self, inputs, state, scope=None):
        wrapped_rnn_state = state[1]
        wrapped_output, wrapped_state = self._wrapped_rnn_cell(
            inputs,
            wrapped_rnn_state,
            scope=scope,
        )

        return wrapped_output, (wrapped_output, wrapped_state)

    @property
    def state_size(self):
        return (
            self._wrapped_rnn_cell.output_size,
            self._wrapped_rnn_cell.state_size,
        )

    @property
    def output_size(self):
        return self._wrapped_rnn_cell.output_size


class OutputStateRnn(tf.contrib.rnn.RNNCell):
    """
    An RNN that wraps another RNN. This RNN outputs the state at every time
    step (in addition to the normal output).
    """

    def __init__(
            self,
            rnn_cell: tf.contrib.rnn.RNNCell,
    ):
        self._wrapped_rnn_cell = rnn_cell

    def __call__(self, inputs, state, scope=None):
        wrapped_rnn_state_size = state
        wrapped_output, wrapped_state = self._wrapped_rnn_cell(
            inputs,
            wrapped_rnn_state_size,
            scope=scope,
        )

        return (wrapped_output, wrapped_state), wrapped_state

    @property
    def state_size(self):
        return self._wrapped_rnn_cell.state_size

    @property
    def output_size(self):
        return (
            self._wrapped_rnn_cell.output_size,
            self._wrapped_rnn_cell.state_size,
        )


class SaveEverythingRnn(tf.contrib.rnn.RNNCell):
    """
    An RNN that wraps another RNN. This RNN outputs the state at every time
    step (in addition to the normal output) and also saves the output to the
    state.
    """

    def __init__(
            self,
            rnn_cell: tf.contrib.rnn.RNNCell,
    ):
        self._wrapped_rnn_cell = rnn_cell

    def __call__(self, inputs, state, scope=None):
        # wrapped_rnn_input = inputs[0]
        wrapped_rnn_state = state[1]
        wrapped_output, wrapped_state = self._wrapped_rnn_cell(
            inputs,
            wrapped_rnn_state,
            scope=scope,
        )

        return (wrapped_output, wrapped_state), (wrapped_output, wrapped_state)

    @property
    def state_size(self):
        return (
            self._wrapped_rnn_cell.output_size,
            self._wrapped_rnn_cell.state_size,
        )

    @property
    def output_size(self):
        return (
            self._wrapped_rnn_cell.output_size,
            self._wrapped_rnn_cell.state_size,
        )


class LNLSTMCell(LSTMCell):
    """
    Same as LSTMCell but add LayerNorm.
    """

    def __call__(self, inputs, state, scope=None):
        """Run one step of LSTM.
    
        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
          scope: VariableScope for the created subgraph; defaults to "lstm_cell".
    
        Returns:
          A tuple containing:
    
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.
    
        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units],
                                     [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError(
                "Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope(scope or "lstm_cell",
                               initializer=self._initializer) as unit_scope:
            if self._num_unit_shards is not None:
                unit_scope.set_partitioner(
                    partitioned_variables.fixed_size_partitioner(
                        self._num_unit_shards))
            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            inputs = layer_normalize(inputs, name="lstm_inputs")
            m_prev = layer_normalize(m_prev, name="lstm_m_prev")
            lstm_matrix = _linear([inputs, m_prev], 4 * self._num_units,
                                  bias=True,
                                  scope=scope)
            i, j, f, o = array_ops.split(
                value=lstm_matrix, num_or_size_splits=4, axis=1)

            # Diagonal connections
            if self._use_peepholes:
                with vs.variable_scope(unit_scope) as projection_scope:
                    if self._num_unit_shards is not None:
                        projection_scope.set_partitioner(None)
                    w_f_diag = vs.get_variable(
                        "w_f_diag", shape=[self._num_units], dtype=dtype)
                    w_i_diag = vs.get_variable(
                        "w_i_diag", shape=[self._num_units], dtype=dtype)
                    w_o_diag = vs.get_variable(
                        "w_o_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (
                sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                     self._activation(j))

            if self._cell_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
                # pylint: enable=invalid-unary-operand-type

            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)

            if self._num_proj is not None:
                with vs.variable_scope("projection") as proj_scope:
                    if self._num_proj_shards is not None:
                        proj_scope.set_partitioner(
                            partitioned_variables.fixed_size_partitioner(
                                self._num_proj_shards))
                    m = _linear(m, self._num_proj, bias=False, scope=scope)

                if self._proj_clip is not None:
                    # pylint: disable=invalid-unary-operand-type
                    m = clip_ops.clip_by_value(m, -self._proj_clip,
                                               self._proj_clip)
                    # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))
        return m, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: (optional) Variable scope to create parameters in.
  
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError(
                "linear expects shape[1] to be provided for shape %s, "
                "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            "weights", [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                "biases", [output_size],
                dtype=dtype,
                initializer=init_ops.constant_initializer(bias_start,
                                                          dtype=dtype))
    return nn_ops.bias_add(res, biases)


class RWACell(tf.contrib.rnn.RNNCell):
    """
    Recurrent weighted averge cell (https://arxiv.org/abs/1703.01253)

    Implementation from
    https://github.com/jostmey/cas/blob/master/RWACell.py?im
    """

    def __init__(self, num_units, decay_rate=0.0):
        """Initialize the RWA cell.
        Args:
            num_units: int, The number of units in the RWA cell.
            decay_rate: (optional) If this is a float it sets the
                decay rate for every unit. If this is a list or
                tensor of shape `[num_units]` it sets the decay
                rate for each individual unit. The decay rate is
                defined as `ln(2.0)/hl` where `hl` is the desired
                half-life of the memory.
        """

        self.num_units = num_units
        if type(
                decay_rate) is not tf.Variable:  # Do nothing if the decay rate is learnable
            decay_rate = tf.convert_to_tensor(decay_rate)
        self.decay_rate = decay_rate
        self.activation = tf.nn.tanh

    def zero_state(self, batch_size, dtype):
        """`zero_state` is overridden to return non-zero values and
        parameters that must be learned."""

        num_units = self.num_units
        activation = self.activation

        n = tf.zeros([batch_size, num_units], dtype=dtype)
        d = tf.zeros([batch_size, num_units], dtype=dtype)
        h = tf.zeros([batch_size, num_units], dtype=dtype)
        a_max = -float('inf') * tf.ones([batch_size, num_units],
                                        dtype=dtype)  # Start off with a large negative number with room for this value to decay

        """The scope for the RWACell is hard-coded into `RWACell.zero_state`.
        This is done because the initial state is learned and some of the model
        parameters must be defined here. These parameters require a scope and
        because `RWACell.zero_state` does not accept the scope as an argument,
        it must be hard-coded.
        """
        with tf.variable_scope('RWACell'):
            s_0 = tf.get_variable('s_0', [num_units],
                                  initializer=tf.random_normal_initializer(
                                      stddev=1.0))
            h += activation(tf.expand_dims(s_0, 0))

        return (n, d, h, a_max)

    def __call__(self, inputs, state, scope=None):
        num_inputs = inputs.get_shape()[1]
        num_units = self.num_units
        decay_rate = self.decay_rate
        activation = self.activation
        x = inputs
        n, d, h, a_max = state
        if scope is not None:
            raise ValueError(
                "The argument `scope` for `RWACell.__call__` is deprecated and "
                "no longer works. The scope is hard-coded to make the initial "
                "state learnable. See `RWACell.zero_state` for more details."
            )

        with tf.variable_scope('RWACell'):
            W_u = tf.get_variable('W_u', [num_inputs, num_units],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_u = tf.get_variable('b_u', [num_units],
                                  initializer=tf.constant_initializer(0.0))
            W_g = tf.get_variable('W_g', [num_inputs + num_units, num_units],
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_g = tf.get_variable('b_g', [num_units],
                                  initializer=tf.constant_initializer(0.0))
            W_a = tf.get_variable('W_a', [num_inputs + num_units, num_units],
                                  initializer=tf.contrib.layers.xavier_initializer())

        xh = tf.concat([x, h], 1)

        u = tf.matmul(x, W_u) + b_u
        g = tf.matmul(xh, W_g) + b_g
        a = tf.matmul(xh,
                      W_a)  # The bias term when factored out of the numerator and denominator cancels and is unnecessary
        z = tf.multiply(u, tf.nn.tanh(g))

        a_decay = a_max - decay_rate
        n_decay = tf.multiply(n, tf.exp(-decay_rate))
        d_decay = tf.multiply(d, tf.exp(-decay_rate))

        a_newmax = tf.maximum(a_decay, a)
        exp_diff = tf.exp(a_max - a_newmax)
        exp_scaled = tf.exp(a - a_newmax)
        n = tf.multiply(n_decay, exp_diff) + tf.multiply(z,
                                                         exp_scaled)  # Numerically stable update of numerator
        d = tf.multiply(d_decay,
                        exp_diff) + exp_scaled  # Numerically stable update of denominator
        h = activation(tf.div(n, d))
        a_max = a_newmax

        next_state = (n, d, h, a_max)
        return h, next_state

    @property
    def output_size(self):
        return self.num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units, self.num_units, self.num_units)
