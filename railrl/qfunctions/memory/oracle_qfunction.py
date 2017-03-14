import tensorflow as tf

from railrl.qfunctions.nn_qfunction import NNQFunction


class _SaveActionRnn(tf.nn.rnn_cell.RNNCell):
    """
    An RNN that wraps another RNN. The main difference is that this saves the last action in state.
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
        return (self._wrapped_rnn_cell.state_size,
                self._wrapped_rnn_cell.output_size)

    @property
    def output_size(self):
        return self._wrapped_rnn_cell.output_size


class OracleQFunction(NNQFunction):
    """
    An oracle QFunction that uses a supervised learning cost to know the true
    gradient out the output w.r.t. the loss.
    """
    def __init__(
            self,
            name_or_scope,
            env,
            target_labels=None,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._ocm_env = env
        if target_labels is None:
            target_labels = tf.placeholder(
                tf.int32,
                shape=[
                    None,
                    self._ocm_env.wrapped_env.action_space.flat_dim,
                ],
                name='oracle_target_labels',
            )
        self.target_labels = target_labels
        super().__init__(
            name_or_scope=name_or_scope,
            create_network_dict=dict(
                target_labels=self.target_labels,
            ),
            **kwargs)

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
            target_labels=None,
    ):
        with tf.variable_scope("oracle_loss"):
            out = self._ocm_env.get_tf_loss(
                observation_input,
                action_input,
                target_labels=target_labels,
            )
        return out

    @property
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
            action_input=self.action_input,
            target_labels=self.target_labels,
        )


class OracleUnrollQFunction(NNQFunction):
    """
    An oracle QFunction that uses a supervised learning cost to know the true
    gradient out the output w.r.t. the loss.
    """
    def __init__(
            self,
            name_or_scope,
            env,
            policy,
            env_obs_dim,
            num_bptt_unrolls,
            max_horizon_length,
            target_labels=None,
            sequence_lengths=None,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._ocm_env = env
        self._policy = policy
        self._rnn_cell = self._policy.rnn_cell
        self._rnn_init_state_ph = self._policy.create_init_state_placeholder()
        self._rnn_cell_scope = self._policy.rnn_cell_scope
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, max_horizon_length - num_bptt_unrolls, env_obs_dim],
            name='rnn_time_inputs',
        )
        self._rnn_init_state_ph = self._policy.create_init_state_placeholder()

        self._rnn_inputs = tf.unpack(self._rnn_inputs_ph, axis=1)

        if sequence_lengths is None:
            sequence_lengths = tf.placeholder(
                tf.int32,
                shape=[None],
                name='sequence_length',
            )
        if target_labels is None:
            target_labels = tf.placeholder(
                tf.int32,
                shape=[
                    None,
                    self._ocm_env.wrapped_env.action_space.flat_dim,
                ],
                name='oracle_target_labels',
            )
        self.sequence_lengths = sequence_lengths
        self.target_labels = target_labels
        super().__init__(
            name_or_scope=name_or_scope,
            create_network_dict=dict(
                target_labels=self.target_labels,
            ),
            **kwargs)

    @property
    def sequence_length_placeholder(self):
        return self.sequence_lengths

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
            target_labels=None,
    ):
        self._rnn_outputs, self._rnn_final_state = tf.nn.rnn(
            self._rnn_cell,
            self._rnn_inputs,
            initial_state=tf.split(1, 2, self._rnn_init_state_ph),
            sequence_length=self.sequence_length_placeholder,
            dtype=tf.float32,
            scope=self._rnn_cell_scope,
        )
        final_actions = self._rnn_final_state[-1]
        with tf.variable_scope("oracle_loss"):
            out = self._ocm_env.get_tf_loss(
                observations=observation_input,
                actions=final_actions,
                target_labels=target_labels,
            )
        return out

    @property
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
            action_input=self.action_input,
            target_labels=self.target_labels,
            sequence_lengths=self.sequence_lengths,
        )
