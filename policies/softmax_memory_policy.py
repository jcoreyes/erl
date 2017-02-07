import numpy as np
import tensorflow as tf

from railrl.core.tf_util import he_uniform_initializer, mlp, linear
from railrl.policies.nn_policy import NNPolicy
from rllab.misc.overrides import overrides


class SoftmaxMemoryPolicy(NNPolicy):
    """
    A policy that outputs two things:
        1. A probability distribution over a set of discrete actions
        2. A memory state vector
    """
    def __init__(
            self,
            name_or_scope,
            env_action_dim,
            memory_dim,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            **kwargs
    ):
        """
        :param name_or_scope: String - Name scope of the policy
        :param env_action_dim: int - Dimension of the environment action
        :param memory_dim:  int - Dimension of the memory state
        :param kwargs:
        """
        self.setup_serialization(locals())
        self._env_action_dim = env_action_dim
        self._memory_dim = memory_dim
        self._observation_hidden_sizes = observation_hidden_sizes
        self._hidden_W_init = hidden_W_init or he_uniform_initializer()
        self._hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self._output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self._output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self._hidden_nonlinearity = hidden_nonlinearity or tf.nn.relu
        self._output_nonlinearity = tf.nn.softmax
        super().__init__(name_or_scope=name_or_scope, **kwargs)
        assert (self._env_action_dim, self._memory_dim) == self.output_dim

    def _create_network_internal(self, observation_input=None):
        assert observation_input is not None
        env_obs, memory_obs = observation_input
        env_obs = self._process_layer(
            env_obs,
            scope_name="env_obs",
        )
        memory_obs = self._process_layer(
            memory_obs,
            scope_name="memory_obs",
        )
        observation_input = tf.concat(1, [env_obs, memory_obs])
        with tf.variable_scope("environment_action"):
            with tf.variable_scope("mlp"):
                observation_output = mlp(
                    observation_input,
                    sum(self.observation_dim),
                    self._observation_hidden_sizes,
                    self._hidden_nonlinearity,
                    W_initializer=self._hidden_W_init,
                    b_initializer=self._hidden_b_init,
                    pre_nonlin_lambda=self._process_layer,
                )
            observation_output = self._process_layer(
                observation_output,
                scope_name="output_preactivations",
            )
            with tf.variable_scope("output"):
                env_action = self._output_nonlinearity(linear(
                    observation_output,
                    self._observation_hidden_sizes[-1],
                    self._env_action_dim,
                    W_initializer=self._output_W_init,
                    b_initializer=self._output_b_init,
                ))
        with tf.variable_scope("memory_state"):
            with tf.variable_scope("mlp"):
                observation_output = mlp(
                    observation_input,
                    sum(self.observation_dim),
                    self._observation_hidden_sizes,
                    self._hidden_nonlinearity,
                    W_initializer=self._hidden_W_init,
                    b_initializer=self._hidden_b_init,
                    pre_nonlin_lambda=self._process_layer,
                )
            observation_output = self._process_layer(
                observation_output,
                scope_name="output_preactivations",
            )
            with tf.variable_scope("output"):
                memory_write_action = self._output_nonlinearity(linear(
                    observation_output,
                    self._observation_hidden_sizes[-1],
                    self._memory_dim,
                    W_initializer=self._output_W_init,
                    b_initializer=self._output_b_init,
                ))
        return env_action, memory_write_action

    @overrides
    def get_action(self, observation):
        new_observation = tuple(
            np.expand_dims(o, axis=0) for o in observation
        )
        return self.sess.run(
            self.output,
            {
                self.observation_input: new_observation,
            }
        ), {}
