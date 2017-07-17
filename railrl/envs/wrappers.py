import gym.spaces
from rllab.core.serializable import Serializable
from rllab.envs.normalized_env import NormalizedEnv
from rllab.envs.proxy_env import ProxyEnv
from sandbox.rocky.tf.spaces import Box as TfBox
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product


class NormalizedBoxEnv(NormalizedEnv):
    @property
    def action_space(self):
        return Box(super().action_space.low,
                   super().action_space.high)

    @property
    def observation_space(self):
        return Box(super().observation_space.low,
                   super().observation_space.high)

    def log_diagnostics(self, paths, **kwargs):
        return self._wrapped_env.log_diagnostics(paths, **kwargs)


normalize = NormalizedBoxEnv


class ConvertEnv(ProxyEnv, Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @property
    def action_space(self):
        return TfBox(super().action_space.low,
                   super().action_space.high)

    @property
    def observation_space(self):
        return TfBox(super().observation_space.low,
                   super().observation_space.high)

    def __str__(self):
        return "TfConverted: %s" % self._wrapped_env

    # def get_param_values(self):
    #     return None
    #
    # def log_diagnostics(self, paths, *args, **kwargs):
    #     if hasattr(self.wrapped_env, "log_diagnostics"):
    #         self.wrapped_env.log_diagnostics(paths, *args, **kwargs)
    #
    # def terminate(self):
    #     pass


convert_to_tf_env = ConvertEnv


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    elif (isinstance(space, Box) or isinstance(space, Discrete)
          or isinstance(space, Product)):
        return space
    else:
        raise NotImplementedError
