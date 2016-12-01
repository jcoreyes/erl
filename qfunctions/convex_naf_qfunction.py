import tensorflow as tf

from policies.argmax_policy import ArgmaxPolicy
from predictors.mlp_state_network import MlpStateNetwork
from qfunctions.action_concave_qfunction import ActionConcaveQFunction
from qfunctions.naf_qfunction import NAFQFunction
from rllab.misc.overrides import overrides


class ConcaveNAF(NAFQFunction):
    def __init__(
            self,
            name_or_scope,
            **kwargs
    ):
        self.setup_serialization(locals())
        self._policy = None
        self._af = None
        self._vf = None
        super(NAFQFunction, self).__init__(
            name_or_scope=name_or_scope,
            **kwargs
        )


    @overrides
    def _create_network(self, observation_input, action_input):
        self._af = ActionConcaveQFunction(
            name_or_scope="advantage_function",
            action_dim=self.action_dim,
            observation_dim=self.observation_dim,
            action_input=action_input,
            observation_input=observation_input,
        )
        self._vf = MlpStateNetwork(
            name_or_scope="V_function",
            output_dim=1,
            observation_dim=self.observation_dim,
            observation_input=observation_input,
            observation_hidden_sizes=(200, 200),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
        )
        return self._vf.output + self._af.output

        # TODO(vpong): subtract max_a A(a, s) to make the advantage function
        # equal the actual advantage function like so:
        # self.policy_output_placeholder = tf.placeholder(
        #     tf.float32,
        #     shape=[None, self.action_dim],
        #     name='policy_output',
        # )
        # with tf.Session() as sess:
        #     # This is a bit ugly, but we need some session to copy the values
        #     # over. The main session should initialize the values again
        #     with sess.as_default():
        #         self.sess.run(tf.initialize_variables(self.get_params()))
        #         self._policy = ArgmaxPolicy(
        #             name_or_scope="argmax_policy",
        #             qfunction=self._af,
        #         )
        #         self.af_copy_with_policy_input = self._af.get_weight_tied_copy(
        #             action_input=self.policy_output_placeholder,
        #             observation_input=observation_input,
        #         )
        # return self._vf.output + (self._af.output -
        #                          self.af_copy_with_policy_input.output)

    @property
    def implicit_policy(self):
        return self.advantage_function.implicit_policy

    @property
    def value_function(self):
        return self._vf

    @property
    def advantage_function(self):
        return self._af


