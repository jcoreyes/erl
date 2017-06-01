from railrl.envs.env_utils import gym_env
from rllab.core.serializable import Serializable
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.gym_env import convert_gym_space
from rllab.envs.normalized_env import normalize
from rllab.envs.proxy_env import ProxyEnv
from sandbox.rocky.tf.spaces import Box


class HiddenCartpoleEnv(CartpoleEnv, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        super().__init__(position_only=True)
        # self.action_space = Box(self.action_space.low, self.action_space.high)
        # self.observation_space = Box(self.observation_space.low,
        #                               self.observation_space.high)
#         # env = CartpoleEnv(position_only=True)
#         # env = gym_env("Pendulum-v0")
#         # import ipdb; ipdb.set_trace()
#         # super().__init__(normalize(env))
#
    @property
    def action_space(self):
        # return self._action_space
        # return convert_gym_space(super().action_space)
        return Box(super().action_space.low,
                                super().action_space.high)
        # self.observation_space = Box(self.observation_space.low,
        #                               self.observation_space.high)

    @property
    def observation_space(self):
        # return convert_gym_space(super().observation_space)
        return Box(super().observation_space.low,
                   super().observation_space.high)
        # return self._observation_space

# class HiddenCartpoleEnv(ProxyEnv):
#     def __init__(self):
#         # env = CartpoleEnv(position_only=True)
#         env = gym_env("Pendulum-v0")
#         # import ipdb; ipdb.set_trace()
#         super().__init__(normalize(env))
#         import ipdb; ipdb.set_trace()
#         self.action_space = Box(self.action_space.low, self.action_space.high)
#         self.observation_space = Box(self.observation_space.low,
#                                       self.observation_space.high)
#
#     @property
#     def action_space(self):
#         return self._action_space
#
#     @property
#     def observation_space(self):
#         return self._observation_space
