import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

from railrl.core import tf_util
from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy


class DecoupledLSTM(LSTMCell):
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
        self.env_action_scope = None

    def __call__(self, inputs, state, scope=None):
        env_action = self.create_env_action(inputs, state)
        write_action = self.create_write_action(env_action, inputs, state)

        return env_action, write_action

    def create_env_action(self, env_obs, memory_obs, reuse=False):
        with tf.variable_scope("env_action", reuse=reuse) as (
                self.env_action_scope
        ):
            flat_inputs = tf.concat(axis=1, values=[env_obs, memory_obs])
            env_action_logit = tf_util.linear(
                flat_inputs,
                flat_inputs.get_shape()[-1],
                self._output_dim,
            )
        return tf.nn.softmax(env_action_logit)

    def create_write_action(self, env_action, env_obs, memory_obs,
                            reuse=False):
        inputs, state = env_obs, memory_obs
        with tf.variable_scope("write_action", reuse=reuse):
            split_state = tf.split(axis=1, num_or_size_splits=2, value=state)
            lstm_input = tf.concat(values=(inputs, env_action), axis=1)
            lstm_output, lstm_state = super().__call__(lstm_input, split_state)
            flat_state = tf.concat(axis=1, values=lstm_state)
        return flat_state

    @property
    def state_size(self):
        return self._num_units * 2

    @property
    def output_size(self):
        return self._output_dim


class ActionAwareMemoryPolicy(RnnCellPolicy):
    """
    logit = function of environment observation and memory
    env action = softmax(logits)
    write = function of environment action and observatino
    """
    def __init__(
            self,
            name_or_scope,
            action_dim,
            memory_dim,
            init_state=None,
            rnn_cell_class=None,
            **kwargs
    ):
        assert memory_dim % 2 == 0
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        print("Ignoring rnn_cell_class")
        self._memory_dim = memory_dim
        self._action_dim = action_dim
        self._rnn_cell = None
        self._rnn_cell_scope = None
        self.init_state = self._placeholder_if_none(
            init_state,
            [None, self._memory_dim],
            name='decoupled_lstm_init_state',
            dtype=tf.float32,
        )
        self.env_actions_ph = tf.placeholder(
            tf.float32,
            [None, action_dim]
        )
        self.env_observation_ph = tf.placeholder(
            tf.float32,
            [None, self.observation_dim[0]],
        )
        self.memory_observation_ph = tf.placeholder(
            tf.float32,
            [None, self._memory_dim]
        )
        self.action_output = None
        self.write_output = None
        self._create_network()

    def _create_network_internal(self, observation_input=None, init_state=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        self._rnn_cell = DecoupledLSTM(
            self._memory_dim,
            self._action_dim,
        )

        with tf.variable_scope("rnn_cell") as self._rnn_cell_scope:
            cell_output = self._rnn_cell(env_obs, memory_obs)

            self.action_output = self._rnn_cell.create_env_action(
                self.env_observation_ph,
                self.memory_observation_ph,
                reuse=True
            )
            self.write_output = self._rnn_cell.create_write_action(
                self.env_actions_ph,
                self.env_observation_ph,
                self.memory_observation_ph,
                reuse=True,
            )

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

    def get_environment_action(self, observation):
        return self.sess.run(
            self.action_output,
            {
                self.env_observation_ph: observation[0:1],
                self.memory_observation_ph: observation[1:2],
            }
        )[0], {}

    def get_write_action(self, env_action, observation):
        return self.sess.run(
            self.write_output,
            {
                self.env_actions_ph: [env_action],
                self.env_observation_ph: observation[0:1],
                self.memory_observation_ph: observation[1:2],
            }
        )[0], {}
