import ray

from railrl.envs.base import RolloutEnv
from railrl.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from railrl.samplers.util import rollout
from railrl.core.serializable import Serializable
import numpy as np
import os

@ray.remote(num_cpus=1)
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
            rollout_function,
    ):
        os.environ["MKL_NUM_THREADS"] = "1"
        self._env = env
        if normalize_env:
            # TODO: support more than just box envs
            self._env = NormalizedBoxEnv(self._env)
        self._policy = policy
        self._exploration_policy = exploration_policy
        self._max_path_length = max_path_length
        self.rollout_function = rollout_function

    def rollout(self, policy_params, use_exploration_strategy):
        self._policy.set_param_values_np(policy_params)
        if use_exploration_strategy:
            policy = self._exploration_policy
        else:
            policy = self._policy
        # return self.multitask_rollout(self._env, policy, self._max_path_length)
        return self.rollout_function(self._env, policy, self._max_path_length)

    #HACK FOR COMPUTING MULITASK ROLLOUTS FOR PARALLEL
    def multitask_rollout(self, env, policy, max_path_length):
        observations = []
        actions = []
        rewards = []
        terminals = []
        agent_infos = []
        env_infos = []
        next_observations = []
        path_length = 0
        o = env.reset()
        goal = env.get_goal()
        while path_length < max_path_length:
            new_obs = np.hstack((o, goal))
            a, agent_info = policy.get_action(new_obs)
            next_o, r, d, env_info = env.step(a)
            observations.append(o)
            rewards.append(r)
            terminals.append(d)
            actions.append(a)
            next_observations.append(next_o)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            path_length += 1
            if d:
                break
            o = next_o

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        return dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1, 1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1, 1),
            agent_infos=agent_infos,
            env_infos=env_infos,
            goals=np.repeat(goal[None], path_length, 0),
        )

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

    It's the responsibility of the caller to call ray.init() at some point before
    initializing an instance of this class. (Okay, this breaks the
    abstraction, but I can't think of a much cleaner alternative for now.)
    """
    def __init__(
            self,
            env,
            policy,
            exploration_policy,
            max_path_length,
            normalize_env,
            rollout_function=rollout,
            instances=6,
            dedicated_train=4,

    ):
        Serializable.quick_init(self, locals())
        super().__init__(env)
        ray.register_custom_serializer(type(env), use_pickle=True)
        ray.register_custom_serializer(type(policy), use_pickle=True)
        ray.register_custom_serializer(type(exploration_policy), use_pickle=True)
        self._ray_envs = [RayEnv.remote(
            env,
            policy,
            exploration_policy,
            max_path_length,
            normalize_env,
            rollout_function,
        ) for _ in range(instances)]
        self._eval_promises = []
        self._train_promises = []

        self.free_envs = set(self._ray_envs)
        self.promise_to_env = {}
        self.num_instances = instances

    def rollout(self, policy, use_exploration_strategy, epoch):
        if len(self._eval_promises) + len(self._train_promises) < self.num_instances:
            self._create_promise(policy, use_exploration_strategy, epoch)
        # Check if remote path has been collected.
        paths, _ = ray.wait(self.promise_list(use_exploration_strategy), timeout=0)
        for path in paths:
            _, path_epoch, path_exploration = self.promise_to_env[path]
            self._free(path)
            if path_epoch != epoch:
                continue
            self._create_promise(policy, use_exploration_strategy, epoch)
            return ray.get(path)
        return None

    def _create_promise(self, policy, use_exploration_strategy, epoch):
        if len(self.promise_list(use_exploration_strategy)) > 1:
            return
        policy_params = policy.get_param_values_np()

        free_env = self._get_free_env()
        promise = free_env.rollout.remote(
            policy_params,
            use_exploration_strategy,
        )
        self.promise_to_env[promise] = (free_env, epoch, use_exploration_strategy)
        self.promise_list(use_exploration_strategy).append(promise)
        return promise

    def _get_free_env(self):
        return self.free_envs.pop()

    def _free(self, promise_id):
        env_id, _, explore = self.promise_to_env[promise_id]
        assert env_id not in self.free_envs
        self.free_envs.add(env_id)
        del self.promise_to_env[promise_id]
        self.promise_list(explore).remove(promise_id)

    def promise_list(self, train_type):
        if train_type:
            return self._train_promises
        return self._eval_promises
