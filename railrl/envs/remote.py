import ray

from railrl.envs.base import RolloutEnv
from railrl.misc.ray_util import set_serialization_mode_to_pickle
from railrl.samplers.util import rollout
from rllab.core.serializable import Serializable
from rllab.envs.normalized_env import normalize
from rllab.envs.proxy_env import ProxyEnv


@ray.remote
class RayEnv(object):
    """
    Perform rollouts asynchronously using ray.
    """
    def __init__(
            self,
            env,
            policy,
            exploration_policy,
            max_path_length,
            normalize_env,
    ):
        self._env = env
        if normalize_env:
            self._env = normalize(self._env)
        self._policy = policy
        self._exploration_policy = exploration_policy
        self._max_path_length = max_path_length

    def rollout(self, policy_params, use_exploration_strategy):
        self._policy.set_param_values_np(policy_params)
        if use_exploration_strategy:
            policy = self._exploration_policy
        else:
            policy = self._policy
        return rollout(self._env, policy, self._max_path_length)


class RemoteRolloutEnv(ProxyEnv, RolloutEnv, Serializable):
    """
    An interface for a rollout environment where the rollouts are performed
    asynchronously.

    This "environment" just talks to the remote environment. The advantage of
    this environment over calling RayEnv directly is that rollout will return
    `None` if a path is not ready, rather than returning a promise (an
    `ObjectID` in Ray-terminology).

    Rather than doing
    ```
    env = CarEnv(foo=1)
    path = env.rollout()  # blocks until rollout is done
    # do some computation
    ```
    you can do
    ```
    remote_env = RemoteRolloutEnv(CarEnv, {'foo': 1})
    path = remote_env.rollout()
    while path is None:
        # do other computation asynchronously
        path = remote_env.rollout() ```
    ```
    So you pass the environment class (CarEnv) and parameters to create the
    environment to RemoteRolloutEnv. What happens under the door is that the
    RemoteRolloutEnv will create its own instance of CarEnv with those
    parameters.

    Note that you could use use RayEnv directly like this:
    ```
    env = CarEnv(foo=1)
    ray_env = RayEnv(CarEnv, {'foo': 1})
    path = ray_env.rollout.remote()

    # Do some computation asyncronously, but eventually call
    path = ray.get(path)  # blocks
    # or
    paths, _ = ray.wait([path])  # polls
    ```
    The main issue is that the caller then has to call `ray` directly, which is
    breaks some abstractions around ray. Plus, then things like
    `ray_env.action_space` wouldn't work
    """
    def __init__(
            self,
            env,
            policy,
            exploration_policy,
            max_path_length,
            normalize_env,
    ):
        Serializable.quick_init(self, locals())
        super().__init__(env)
        set_serialization_mode_to_pickle(type(env))
        set_serialization_mode_to_pickle(type(policy))
        set_serialization_mode_to_pickle(type(exploration_policy))
        self._ray_env = RayEnv.remote(
            env,
            policy,
            exploration_policy,
            max_path_length,
            normalize_env,
        )
        self._rollout_promise = None

    def rollout(self, policy, use_exploration_strategy):
        if self._rollout_promise is None:
            policy_params = policy.get_param_values_np()
            self._rollout_promise = self._ray_env.rollout.remote(
                policy_params,
                use_exploration_strategy,
            )
            return None

        # Check if remote path has been collected.
        paths, _ = ray.wait([self._rollout_promise], timeout=0)

        if len(paths):
            policy_params = policy.get_param_values_np()
            self._rollout_promise = self._ray_env.rollout.remote(
                policy_params,
                use_exploration_strategy,
            )
            return ray.get(paths[0])

        return None
