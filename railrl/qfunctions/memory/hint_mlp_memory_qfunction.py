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
            hint_input=None,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(100,),
            observation_hidden_sizes=(100,),
            hidden_nonlinearity=tf.nn.relu,
            **kwargs
    ):
        self.setup_serialization(locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity
        self.hint_dim = hint_dim
        if hint_input is None:
            hint_input = tf.placeholder(
                tf.float32,
                shape=[
                    None,
                    self.hint_dim,
                ],
                name='hint_input',
            )
        self.hint_input = hint_input
        super().__init__(
            name_or_scope=name_or_scope,
            create_network_dict=dict(
                hint_input=self.hint_input,
            ),
            **kwargs
        )

    def _create_network_internal(
            self,
            observation_input=None,
            action_input=None,
            hint_input=None
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
        observation_input = tf.concat(1, [env_obs, memory_obs, hint_input])
        action_input = tf.concat(1, [env_action, memory_action])
        with tf.variable_scope("observation_mlp"):
            observation_output = mlp(
                observation_input,
                sum(self.observation_dim) + self.hint_dim,
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
        embedded = tf.concat(1, [observation_output, action_input])
        embedded_dim = sum(self.action_dim) + self.observation_hidden_sizes[-1]
        with tf.variable_scope("fusion_mlp"):
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

        with tf.variable_scope("output_linear"):
            return linear(
                fused_output,
                self.embedded_hidden_sizes[-1],
                1,
                W_initializer=self.output_W_init,
                b_initializer=self.output_b_init,
            )

    @property
    def _input_name_to_values(self):
        return dict(
            observation_input=self.observation_input,
            action_input=self.action_input,
            hint_input=self.hint_input,
        )
