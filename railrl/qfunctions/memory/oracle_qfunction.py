import tensorflow as tf

from railrl.core.tf_util import he_uniform_initializer, mlp, linear
from railrl.envs.memory.one_char_memory import OneCharMemory
from railrl.qfunctions.nn_qfunction import NNQFunction


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
        self._sequence_length = tf.placeholder(
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
        self.target_labels = target_labels
        super().__init__(
            name_or_scope=name_or_scope,
            create_network_dict=dict(
                target_labels=self.target_labels,
            ),
            **kwargs)

    @property
    def sequence_length_placeholder(self):
        return self._sequence_length

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
