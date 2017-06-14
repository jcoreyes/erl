from rllab.core.serializable import Serializable
from rllab.envs.normalized_env import NormalizedEnv
from rllab.envs.proxy_env import ProxyEnv
from sandbox.rocky.tf.spaces import Box


class NormalizedTfEnv(NormalizedEnv):
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


normalize_tf = NormalizedTfEnv


class ConvertEnv(ProxyEnv, Serializable):
    def __init__(self, env):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    @property
    def action_space(self):
        return Box(super().action_space.low,
                   super().action_space.high)

    @property
    def observation_space(self):
        return Box(super().observation_space.low,
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