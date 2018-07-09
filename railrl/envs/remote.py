import ray

from railrl.envs.base import RolloutEnv
from railrl.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from railrl.core.serializable import Serializable
import numpy as np
import os
import redis
import torch
import railrl.torch.pytorch_util as ptu

called_ray_init = False
def try_init_ray():
    global called_ray_init
    if not called_ray_init:
        try:
            ray.init(redis_address="127.0.0.1:6379")
        except (redis.exceptions.ConnectionError, AssertionError):
            raise AssertionError(
                "Make sure to call \
                'ray start --head --node-ip-address 127.0.0.1 --redis-port 6379 --redis-shard-ports 6380' \
                beforehand"
            )
        called_ray_init = True

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
            train_rollout_function,
            eval_rollout_function,
    ):
        torch.set_num_threads(1)
        self._env = env
        if normalize_env:
            # TODO: support more than just box envs
            self._env = NormalizedBoxEnv(self._env)
        self._policy = policy
        self._exploration_policy = exploration_policy
        self._max_path_length = max_path_length
        self.train_rollout_function = train_rollout_function
        self.eval_rollout_function = eval_rollout_function

    def rollout(self, policy_params, use_exploration_strategy):
        if use_exploration_strategy:
            self._exploration_policy.set_param_values_np(policy_params)
            policy = self._exploration_policy
            rollout_function = self.train_rollout_function
            if hasattr(self._env, 'train'):
                self._env.train()
        else:
            self._policy.set_param_values_np(policy_params)
            policy = self._policy
            rollout_function = self.eval_rollout_function
            if hasattr(self._env, 'eval'):
                self._env.eval()

        rollout = rollout_function(
            self._env,
            policy,
            self._max_path_length
        )
        if 'full_observations' in rollout:
            rollout['observations'] = rollout['full_observations'][:-1]
        return rollout

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
            train_rollout_function,
            eval_rollout_function,
            num_workers=2,
            dedicated_train=1,
    ):
        Serializable.quick_init(self, locals())
        super().__init__(env)
        ray.register_custom_serializer(type(env), use_pickle=True)
        ray.register_custom_serializer(type(policy), use_pickle=True)
        ray.register_custom_serializer(type(exploration_policy), use_pickle=True)
        self._ray_envs = [
            #ray.remote(num_gpus=1 if ptu.gpu_enabled() else 0)(RayEnv).remote(
            RayEnv.remote(
                env,
                policy,
                exploration_policy,
                max_path_length,
                normalize_env,
                train_rollout_function,
                eval_rollout_function,
            )
        for _ in range(num_workers)]

        self.free_envs = set(self._ray_envs)
        self.promise_info = {}
        # Let self.promise_list[True] be the promises for training
        # and self.promise_list[False] be the promises for eval.
        self.promise_list = {
            True: [],
            False: [],
        }

        self.num_workers = num_workers
        # Let self.worker_limits[True] be the max number of workers for training
        # and self.worker_limits[False] be the max number of workers for eval.
        self.worker_limits = {
            True: dedicated_train,
            False: self.num_workers - dedicated_train,
        }

    def rollout(self, policy, train, epoch):
        if len(self.promise_list[train]) < self.worker_limits[train]:
            self._alloc_promise(policy, train, epoch)

        # Check if remote path has been collected.
        ready_promises, _ = ray.wait(self.promise_list[train], timeout=0)
        for promise in ready_promises:
            _, path_epoch, path_type = self.promise_info[promise]
            self._free_promise(promise)
            # Throw away eval paths from previous epochs
            if path_epoch != epoch and not train:
                continue
            self._alloc_promise(policy, train, epoch)
            return ray.get(promise)
        return None

    def _alloc_promise(self, policy, train, epoch):
        policy_params = policy.get_param_values_np()

        free_env = self.free_envs.pop()
        promise = free_env.rollout.remote(
            policy_params,
            train,
        )
        self.promise_info[promise] = (free_env, epoch, train)
        self.promise_list[train].append(promise)
        return promise

    def _free_promise(self, promise_id):
        env_id, _, train = self.promise_info[promise_id]
        assert env_id not in self.free_envs
        self.free_envs.add(env_id)
        del self.promise_info[promise_id]
        self.promise_list[train].remove(promise_id)
