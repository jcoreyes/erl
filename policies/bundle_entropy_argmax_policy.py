from optimizers.bundle_entropy import solveBatch
from rllab.core.serializable import Serializable


class BundleEntropyArgmaxPolicy(object):
    """
    A policy that outputs

    pi(s) = argmax_a Q(a, s)

    The policy is optimized using the bundle entropy method described in [1].

    References
    ----------
    .. [1] Amos, Brandon, Lei Xu, and J. Zico Kolter.
           "Input Convex Neural Networks." arXiv preprint arXiv:1609.07152 (2016).
    """
    def __init__(
        self,
        qfunction,
        action_dim,
        learning_rate=1e-1,
        n_update_steps=50,
    ):
        """

        :param name_or_scope:
        :param qfunction: Some NNQFunction
        :param action_dim:
        :param learning_rate: Learning rate.
        :param n_update_steps: How many optimization steps to take to figure out
        the action.
        """
        Serializable.quick_init(self, locals())
        self.qfunction = qfunction
        self.learning_rate = learning_rate
        self.n_update_steps = n_update_steps

        self.observation_input = qfunction.observation_input
        self.action_dim = qfunction.action_dim
        self.observation_dim = qfunction.observation_dim

    def get_action(self, observation):
        debug_dict = {}
        action = None
        return action, debug_dict


    def act(self, test=False):
        obs = np.expand_dims(self.observation, axis=0)
        if FLAGS.use_gd:
            act = self.get_cvx_opt_gd(self._opt_test_gd, obs)
        else:
            act = self.get_cvx_opt(self._opt_test, obs)
        action = act if test else self._act_expl(act)
        action = np.clip(action, -1, 1)
        self.action = np.atleast_1d(np.squeeze(action, axis=0))  # TODO: remove this hack
        return self.action

    def get_cvx_opt(self, func, obs):
        act = np.ones((obs.shape[0], self.dimA)) * 0.5
        def fg(x):
            value, grad = func(obs, 2 * x - 1)
            grad *= 2
            return value, grad

        act = bundle_entropy.solveBatch(fg, act)[0]
        act = 2 * act - 1

        return act
