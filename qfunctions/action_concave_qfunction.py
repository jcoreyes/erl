import tensorflow as tf
from railrl.policies.argmax_policy import ArgmaxPolicy
from railrl.policies.bundle_entropy_argmax_policy import BundleEntropyArgmaxPolicy
from railrl.qfunctions.nn_qfunction import NNQFunction

from railrl.core.tf_util import linear, he_uniform_initializer, weight_variable, \
    bias_variable
from railrl.qfunctions.optimizable_q_function import OptimizableQFunction


class ActionConcaveQFunction(NNQFunction, OptimizableQFunction):
    """
    Action Concave Q function used in the ICNN paper.
    """
    def __init__(
            self,
            name_or_scope,
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            embedded_hidden_sizes=(200, 200),
            observation_hidden_sizes=(200, 200),
            optimizer_type='sgd',
            **kwargs
    ):
        self.setup_serialization(locals())
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(1.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.embedded_hidden_sizes = embedded_hidden_sizes
        self.observation_hidden_sizes = observation_hidden_sizes
        self.optimizer_type = optimizer_type
        self.hidden_nonlinearity = tf.nn.relu
        self._policy = None
        self._weights_to_clip = []
        super().__init__(name_or_scope=name_or_scope, **kwargs)

    def _create_network(self, observation_input, action_input):
        """
        Let
            a = action
            o = observation
            g = ReLU function
            W = matrix
            b = bias

        u = "observation path"
        u1 = o
        u2 = g(Wo_1 u1 + bo_1)

        z = "action path"
        z1 = a
        z2 = g(
            Wz_1 (z1 * [ Wzu_1 u1 + bz_1]) +
            Wy_1 (y * [ Wyu_1 u1 + by_1]) +
            Wu_1 u1 + b1
        ))
        z3 = g(
            Wz_2 (z2 * [ Wzu_2 u2 + bz_2]) +
            Wy_2 (y * [ Wyu_2 u2 + by_2]) +
            Wu_2 u2 + b2
        ))
        output = -z3

        :param observation_input:
        :param action_input:
        :return:
        """
        z1 = action_input
        z1_size = self.action_dim
        z2_size = 200
        z3_size = 200
        z4_size = 1
        u1 = observation_input
        u1_shape = self.observation_dim
        u2_shape = 200
        u3_shape = 200

        with tf.variable_scope("observation_mlp") as _:
            with tf.variable_scope("u2") as _:
                u2 = tf.nn.relu(linear(u1, u1_shape, u2_shape))
            with tf.variable_scope("u3") as _:
                u3 = tf.nn.relu(linear(u2, u2_shape, u3_shape))

        with tf.variable_scope("z"):
            z2 = self.create_z(action_input, self.action_dim, 1, u1, u1_shape,
                               z1, z1_size, z2_size)
            z3 = self.create_z(action_input, self.action_dim, 2, u2, u2_shape,
                               z2, z2_size, z3_size)
            z4 = self.create_z(action_input, self.action_dim, 3, u3, u3_shape,
                               z3, z3_size, z4_size)
        return -z4

    @property
    def weights_to_clip(self):
        """
        Params that should be clipped to non-negative
        """
        return self._weights_to_clip

    @property
    def implicit_policy(self):
        if self._policy is None:
            self._sgd_policy = ArgmaxPolicy(
                name_or_scope="argmax_policy",
                qfunction=self,
            )
            self._bundle_policy = BundleEntropyArgmaxPolicy(
                qfunction=self,
                action_dim=self.action_dim,
                sess=self.sess,
            )
            with self.sess.as_default():
                self.sess.run(tf.initialize_variables(self.get_params()))
                # TODO(vpong): pass in the optimizer
                if self.optimizer_type == 'sgd':
                    print("Making SGD optimizer")
                    self._policy = ArgmaxPolicy(
                        name_or_scope="argmax_policy",
                        qfunction=self,
                    )
                elif self.optimizer_type == 'bundle_entropy':
                    print("Making bundle optimizer")
                    self._policy = BundleEntropyArgmaxPolicy(
                        qfunction=self,
                        action_dim=self.action_dim,
                        sess=self.sess,
                    )
                else:
                    raise Exception(
                        "Optimizer_type not recognized: {0}".format(
                            self.optimizer_type))
        return self._policy

    def create_z(self, y, y_shape, i, ui, ui_shape, z_last, last_zi_size,
                 next_zi_size):
        Wz_i = weight_variable(
            (last_zi_size, next_zi_size),
            name='Wz_{0}'.format(i),
        )
        self._weights_to_clip.append(Wz_i)
        bz_i = bias_variable(
            (1, last_zi_size),
            name='bz_{0}'.format(i),
        )
        Wzu_i = weight_variable(
            (ui_shape, last_zi_size),
            name='Wzu_{0}'.format(i),
        )

        Wy_i = weight_variable(
            (y_shape, next_zi_size),
            name='Wy_{0}'.format(i),
        )
        by_i = bias_variable(
            (1, y_shape),
            name='by_{0}'.format(i),
        )
        Wyu_i = weight_variable(
            (ui_shape, y_shape),
            name='Wyu_{0}'.format(i),
        )

        Wu_i = weight_variable(
            (ui_shape, next_zi_size),
            name='Wu_{0}'.format(i),
        )
        b_i = bias_variable(
            (1, next_zi_size),
            name='b_{0}'.format(i),
        )
        return tf.nn.relu(
            tf.matmul(
                tf.mul(
                    z_last,
                    tf.matmul(ui, Wzu_i) + bz_i
                ),
                Wz_i) +
            tf.matmul(
                tf.mul(
                    y,
                    tf.matmul(ui, Wyu_i) + by_i
                ),
                Wy_i) +
            tf.matmul(ui, Wu_i) + b_i
        )
