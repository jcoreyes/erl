import tensorflow as tf

from railrl.core.tf_util import weight_variable
from railrl.policies.nn_policy import NNPolicy


class SumPolicy(NNPolicy):
    """Just output the sum of the inputs. This is used to debug."""
    def __init__(self, *args, **kwargs):
        self.setup_serialization(locals())
        super().__init__(*args, **kwargs)
        self._create_network()

    def _create_network_internal(self, observation_input):
        W_obs = weight_variable((self.observation_dim, 1),
                                initializer=tf.constant_initializer(1.))
        return tf.matmul(observation_input, W_obs)
