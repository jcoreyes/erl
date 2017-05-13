import tensorflow as tf

from railrl.core.tf_util import he_uniform_initializer, mlp, linear
from railrl.qfunctions.nn_qfunction import NNQFunction


class MlpMemoryQFunction(NNQFunction):
    def __init__(
            self,
            name_or_scope,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
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
        self._create_network()

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
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
        observation_input = tf.concat(axis=1, values=[env_obs, memory_obs])
        action_input = tf.concat(axis=1, values=[env_action, memory_action])
        obs_input_dim = sum(self.observation_dim)
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
