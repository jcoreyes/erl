import tensorflow as tf

from railrl.core.rnn.rnn import SaveOutputRnn
from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.qfunctions.nn_qfunction import NNQFunction


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
            **kwargs
    ):
        self.setup_serialization(locals())
        self._ocm_env = env
        self._policy = policy
        self._save_rnn_cell = SaveOutputRnn(
            self._policy.rnn_cell,
        )
        self._rnn_init_state_ph = self._policy.get_init_state_placeholder()
        self._rnn_cell_scope = self._policy.rnn_cell_scope

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
        self.save_rnn_inputs = save_rnn_inputs
        self.sequence_lengths = sequence_lengths
        self.target_labels = target_labels
        super().__init__(
            name_or_scope=name_or_scope,
            create_network_dict=dict(
                target_labels=self.target_labels,
                sequence_lengths=self.sequence_lengths,
                save_rnn_inputs=self.save_rnn_inputs,
            ),
            **kwargs)

    @property
    def sequence_length_placeholder(self):
        return self.sequence_lengths

    @property
    def rest_of_obs_placeholder(self):
        return self.save_rnn_inputs

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
            target_labels=None,
            sequence_lengths=None,
            save_rnn_inputs=None,
    ):
        rnn_inputs = tf.unstack(save_rnn_inputs, axis=1)
        self._rnn_cell_scope.reuse_variables()
        init_state = (action_input[1], action_input[0])
        if save_rnn_inputs.get_shape()[1] == 0:
            # In this case, there' no unrolling.
            final_actions = action_input
        else:
            self._rnn_outputs, self._rnn_final_state = tf.nn.rnn(
                self._save_rnn_cell,
                rnn_inputs,
                initial_state=init_state,
                sequence_length=sequence_lengths,
                dtype=tf.float32,
                scope=self._rnn_cell_scope,
            )
            # The action still needs to be in the original shape, so it needs to
            # have the added memory.
            final_actions = (self._rnn_final_state[1], self._rnn_final_state[0])
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
        )