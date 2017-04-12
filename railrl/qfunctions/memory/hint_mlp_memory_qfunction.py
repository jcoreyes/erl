import tensorflow as tf

from railrl.core.tf_util import he_uniform_initializer, mlp, linear
from railrl.qfunctions.nn_qfunction import NNQFunction


class HintMlpMemoryQFunction(NNQFunction):
    """
    Same as MlpMemoryQFunction, except that this critic receives the true 
    target as input.
    """
    def __init__(
            self,
            name_or_scope,
            hint_dim,
            target_labels=None,
            time_labels=None,
            max_time=None,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            use_time=False,
            **kwargs
    ):
        self.setup_serialization(locals())
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.hint_dim = hint_dim
        self.target_labels = self._placeholder_if_none(
            target_labels,
            shape=[None, self.hint_dim],
            name='target_labels',
            dtype=tf.float32,
        )
        self.max_time = max_time
        self.time_labels = self._placeholder_if_none(
            time_labels,
            shape=[None],
            name='time_labels',
            dtype=tf.int32,
        )
        self.use_time = use_time
        self._create_network()

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
            target_labels=None,
            time_labels=None,
    ):
        env_obs, memory_obs = observation_input
        env_action, memory_action = action_input
        env_obs = self._process_layer(
            env_obs,
            scope_name="env_obs",
        )
        memory_obs = self._process_layer(
            memory_obs,
            scope_name="memory_obs",
        )
        env_action = self._process_layer(
            env_action,
            scope_name="env_action",
        )
        memory_action = self._process_layer(
            memory_action,
            scope_name="memory_action",
        )
        if self.use_time:
            time_onehots = tf.one_hot(indices=time_labels,
                                      depth=self.max_time,
                                      on_value=1.0,
                                      off_value=0., name="time_onehots")
            observation_input = tf.concat(
                axis=1,
                values=[env_obs, memory_obs, target_labels, time_onehots],
            )
            obs_input_dim = (
                sum(self.observation_dim) + self.hint_dim + self.max_time
            )
        else:
            observation_input = tf.concat(
                axis=1,
                values=[env_obs, memory_obs, target_labels],
            )
            obs_input_dim = sum(self.observation_dim) + self.hint_dim
        action_input = tf.concat(
            axis=1,
            values=[env_action, memory_action],
        )
        with tf.variable_scope("observation_mlp"):
            if len(self.observation_hidden_sizes) > 0:
                observation_output = mlp(
                    observation_input,
                    obs_input_dim,
                    self.observation_hidden_sizes,
                    self.hidden_nonlinearity,
                    W_initializer=self.hidden_W_init,
                    b_initializer=self.hidden_b_init,
                    pre_nonlin_lambda=self._process_layer,
                )
                observation_output = self._process_layer(
                    observation_output,
                    scope_name="observation_output",
                )
                obs_output_dim = self.observation_hidden_sizes[-1]
            else:
                observation_output = observation_input
                obs_output_dim = obs_input_dim
        embedded = tf.concat(axis=1, values=[observation_output, action_input])
        embedded_dim = sum(self.action_dim) + obs_output_dim
        with tf.variable_scope("fusion_mlp"):
            if len(self.embedded_hidden_sizes) > 0:
                fused_output = mlp(
                    embedded,
                    embedded_dim,
                    self.embedded_hidden_sizes,
                    self.hidden_nonlinearity,
                    W_initializer=self.hidden_W_init,
                    b_initializer=self.hidden_b_init,
                    pre_nonlin_lambda=self._process_layer,
                )
                fused_output = self._process_layer(fused_output)
                fused_output_dim = self.embedded_hidden_sizes[-1]
            else:
                fused_output = embedded
                fused_output_dim = embedded_dim

        with tf.variable_scope("output_linear"):
            return self.output_nonlinearity(linear(
                fused_output,
                fused_output_dim,
                1,
                W_initializer=self.output_W_init,
                b_initializer=self.output_b_init,
            ))

    @property
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
            action_input=self.action_input,
            target_labels=self.target_labels,
            time_labels=self.time_labels,
        )
