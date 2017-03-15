import tensorflow as tf

from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.qfunctions.nn_qfunction import NNQFunction


class _SaveActionRnn(tf.nn.rnn_cell.RNNCell):
    """
    An RNN that wraps another RNN. The main difference is that this saves the
    last action in state.
    """
    def __init__(
            self,
            rnn_cell: tf.nn.rnn_cell.RNNCell,
            action_dim: int,
    ):
        self._wrapped_rnn_cell = rnn_cell
        self._action_dim = action_dim

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
        return self._wrapped_rnn_cell.state_size, self._action_dim


    @property
    def output_size(self):
        return self._action_dim


class OracleUnrollQFunction(NNQFunction):
    """
    An oracle QFunction that uses a supervised learning cost to know the true
    gradient out the output w.r.t. the loss.
    """
    def __init__(
            self,
            name_or_scope,
            env,
            policy: RnnCellPolicy,
            env_obs_dim,
            env_action_dim,
            num_bptt_unrolls,
            max_horizon_length,
            target_labels=None,
            sequence_lengths=None,
            save_rnn_inputs=None,
            save_rnn_init_state=None,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._ocm_env = env
        self._policy = policy
        self._save_rnn_cell = _SaveActionRnn(
            self._policy.rnn_cell,
            env_action_dim
        )
        self._rnn_init_state_ph = self._policy.get_init_state_placeholder()
        self._rnn_cell_scope = self._policy.rnn_cell_scope

        self._ignored_init_last_action_state = tf.placeholder(
            tf.float32,
            shape=[None, env_action_dim],
            name="ignored_init_last_action_state",
        )

        if save_rnn_init_state is None:
            save_rnn_init_state = (
                self._policy.get_init_state_placeholder(),
                self._ignored_init_last_action_state,
            )
        if save_rnn_inputs is None:
            save_rnn_inputs = tf.placeholder(
                tf.float32,
                [None, max_horizon_length - num_bptt_unrolls, env_obs_dim],
                name='rnn_time_inputs',
            )
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
        self.save_rnn_init_state = save_rnn_init_state
        self.save_rnn_inputs = save_rnn_inputs
        self.sequence_lengths = sequence_lengths
        self.target_labels = target_labels
        super().__init__(
            name_or_scope=name_or_scope,
            create_network_dict=dict(
                target_labels=self.target_labels,
                sequence_lengths=self.sequence_lengths,
                save_rnn_inputs=self.save_rnn_inputs,
                save_rnn_init_state=self.save_rnn_init_state,
            ),
            **kwargs)

    @property
    def sequence_length_placeholder(self):
        return self.sequence_lengths

    @property
    def rest_of_obs_placeholder(self):
        return self.save_rnn_inputs

    @property
    def ignored_init_last_action_state(self):
        return self._ignored_init_last_action_state

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
            target_labels=None,
            sequence_lengths=None,
            save_rnn_inputs=None,
            save_rnn_init_state=None,
    ):
        rnn_inputs = tf.unpack(save_rnn_inputs, axis=1)
        self._rnn_cell_scope.reuse_variables()
        self._rnn_outputs, self._rnn_final_state = tf.nn.rnn(
            self._save_rnn_cell,
            rnn_inputs,
            initial_state=save_rnn_init_state,
            sequence_length=sequence_lengths,
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
            save_rnn_inputs=self.save_rnn_inputs,
            save_rnn_init_state=self.save_rnn_init_state,
        )