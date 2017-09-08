import abc
import ray

from railrl.samplers.util import rollout
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv


class RolloutEnv(object):
    """ Environment that support only full rollouts. """

    @abc.abstractmethod
    def rollout(self, *args, **kwargs):
        """
        Non-blocking method for doing rollouts.

        :param args:
        :param kwargs:
        :return: None if no complete paths has been collected.
        Otherwise return a path.
        """
        pass


@ray.remote
class RayEnv(object):
    def __init__(
            self,
            env_class,
            env_params,
            policy_class,
            policy_params,
            # exploration_strategy_class,
            # exploration_strategy_params,
            max_path_length,
    ):
        self._env = env_class(**env_params)
        self._policy = policy_class(**policy_params)
        # TODO
        # self._es = exploration_strategy_class(**exploration_strategy_params)
        self._max_path_length = max_path_length

    def rollout(self, policy_params):
        self._policy.set_param_values_np(policy_params)
        return rollout(self._env, self._policy, self._max_path_length)


class RemoteRolloutEnv(ProxyEnv, RolloutEnv, Serializable):
    """
    A synchronous interface for a remote rollout environment.

    This "environment" basically just talkst o the remote environment.
    The main difference is that rollout will return None if the a path is not
    ready, rather than
    """
    def __init__(
            self,
            env_class,
            env_params,
            *ray_env_args,
            **ray_env_kwargs
    ):
        Serializable.quick_init(self, locals())
        super().__init__(env_class(**env_params))
        ray.init()
        self._ray_env_id = RayEnv.remote(
            env_class,
            env_params,
            *ray_env_args,
            **ray_env_kwargs
        )
        self._rollout_promise = None

    def rollout(self, policy):
        if self._rollout_promise is None:
            policy_params = policy.get_param_values_np()
            self._rollout_promise = self._ray_env_id.rollout.remote(
                policy_params
            )
            return None

        # Check if remote path has been collected.
        paths, _ = ray.wait([self._rollout_promise], timeout=0)

        if len(paths):
            policy_params = policy.get_param_values_np()
            self._rollout_promise = self._ray_env_id.rollout.remote(
                policy_params
            )
            return ray.get(paths[0])

        return None
