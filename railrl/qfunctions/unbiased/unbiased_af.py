import tensorflow as tf

from railrl.predictors.mlp_state_network import MlpStateNetwork
from railrl.qfunctions.nn_qfunction import NNQFunction
from railrl.core import tf_util
from railrl.qfunctions.optimizable_q_function import OptimizableQFunction


class UnbiasedAf(NNQFunction, OptimizableQFunction):
    """
    Given a policy pi parameterized by theta, represent the advantage
    function as

        Q(s, a) = -0.5 (a - pi(s))^T P(s) (a - pi(s))
                   + (a - pi(s))^T \grad_theta pi(s)^T w

    where

        P(s) = L(s) L(s)^T

    and L(s) is a lower triangular matrix. L is parameterized by a
    feedforward neural networks.
    """
    def __init__(
            self,
            name_or_scope,
            policy,
            observation_input=None,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._policy = policy
        if observation_input is None:
            observation_input = self._policy.observation_input
        super().__init__(
            name_or_scope=name_or_scope,
            observation_input=observation_input,
            **kwargs
        )

    def _create_network_internal(self, observation_input, action_input):
        observation_input = self._process_layer(observation_input,
                                                scope_name="observation_input")
        action_input = self._process_layer(action_input,
                                           scope_name="action_input")
        self._L_computer = MlpStateNetwork(
            name_or_scope="L_computer",
            output_dim=self.action_dim * self.action_dim,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(100, 100),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            batch_norm_config=self._batch_norm_config,
        )
        L_output = self._add_subnetwork_and_get_output(self._L_computer)
        L_output = self._process_layer(L_output)
        # L_shape = batch:dimA:dimA
        L = tf_util.vec2lower_triangle(L_output, self.action_dim)
        self.L = L

        delta = action_input - self._policy.output
        h1 = tf.expand_dims(delta, 1)  # h1_shape = batch:1:dimA
        h1 = tf.batch_matmul(h1, L)    # h1_shape = batch:1:dimA
        h1 = tf.batch_matmul(
            h1,
            h1,
            adj_y=True,  # Compute h1 * h1^T
        )                              # h1_shape = batch:1:1
        h1 = tf.squeeze(h1, [1])       # h1_shape = batch:1
        quadratic_term = -0.5 * h1

        policy_vars = self._policy.get_params()
        all_grads = []
        import ipdb
        ipdb.set_trace()
        for policy_var in policy_vars:
            action_grads = [
                tf.gradients(a, policy_var)[0]
                for a in tf.unstack(self._policy.output, axis=1)
            ]
            grads2 = tf.concat(1, action_grads)
            grads = tf.stack(action_grads, axis=1)
            all_grads.append(grads)
        gradients_flat = tf.concat(
            0,
            all_grads,
        )
        linear_params = tf_util.weight_variable(
            gradients_flat.get_shape(),
            name='linear_params',
        )
        linear_term = tf.batch_matmul(
            tf.batch_matmul(
                delta,
                gradients_flat,
                adj_x=True,
            ),
            linear_params,
            adj_x=True,
        )
        return quadratic_term + linear_term

    @property
    def implicit_policy(self):
        return self._policy
