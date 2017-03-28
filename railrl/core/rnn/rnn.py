import tensorflow as tf


class SaveOutputRnn(tf.nn.rnn_cell.RNNCell):
    """
    An RNN that wraps another RNN. This RNN saves the last output in the
    state (in addition to the normal state).
    """
    def __init__(
            self,
            rnn_cell: tf.nn.rnn_cell.RNNCell,
    ):
        self._wrapped_rnn_cell = rnn_cell

    def __call__(self, inputs, state, scope=None):
        wrapped_rnn_state_size = state[0]
        wrapped_output, wrapped_state = self._wrapped_rnn_cell(
            inputs,
            wrapped_rnn_state_size,
            scope=scope,
        )

        return wrapped_output, (wrapped_state, wrapped_output)

    @property
    def state_size(self):
        return (
            self._wrapped_rnn_cell.state_size,
            self._wrapped_rnn_cell.output_size,
        )

    @property
    def output_size(self):
        return self._wrapped_rnn_cell.output_size


class OutputStateRnn(tf.nn.rnn_cell.RNNCell):
    """
    An RNN that wraps another RNN. This RNN outputs the state at every time
    step (in addition to the normal output).
    """
    def __init__(
            self,
            rnn_cell: tf.nn.rnn_cell.RNNCell,
    ):
        self._wrapped_rnn_cell = rnn_cell

    def __call__(self, inputs, state, scope=None):
        wrapped_rnn_state_size = state[0]
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
