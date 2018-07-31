import ray

from railrl.envs.base import RolloutEnv
from railrl.envs.wrappers import NormalizedBoxEnv, ProxyEnv
from railrl.core.serializable import Serializable
import numpy as np
import torch
import railrl.torch.pytorch_util as ptu
import math
from torch.multiprocessing import Process, Pipe
from multiprocessing.connection import wait
import torch.multiprocessing as mp
import railrl.envs.env_utils as env_utils
import cloudpickle

class RemoteRolloutEnv(ProxyEnv, RolloutEnv, Serializable):
    """
    An interface for a rollout environment where the rollouts are performed
    asynchronously.

    This "environment" just talks to the remote environment. The advantage of
    this environment over calling WorkerEnv directly is that rollout will return
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

    This Env can be considered the master env, collecting and managing the
    rollouts of worker envs under the hood. Communication between workers and
    master env are achieved through pipes. No explicit synchronization is
    necessary as the worker/master will alternate between listening and sending
    data through the pipe. Thus, there should never be a case where master and
    worker are both sending or listening.

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
    ):
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self.parent_pipes = []
        self.child_pipes = []

        for _ in range(num_workers):
            parent_conn, child_conn = Pipe()
            self.parent_pipes.append(parent_conn)
            self.child_pipes.append(child_conn)

        self._parallel_envs = [
            Process(
                target=RemoteRolloutEnv._worker_loop,
                args=(
                    self.child_pipes[i],
                    env,
                    policy,
                    exploration_policy,
                    max_path_length,
                    normalize_env,
                    cloudpickle.dumps(train_rollout_function),
                    cloudpickle.dumps(eval_rollout_function),
                )
            )
        for i in range(num_workers)]

        for env in self._parallel_envs:
            env.start()

        self.free_envs = set(self.parent_pipes)
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
            True: math.ceil(self.num_workers / 2),
            False: math.ceil(self.num_workers / 2),
        }

    def rollout(self, policy, train, epoch, discard_other=False):
        # prevent starvation if only one worker
        if discard_other:
            self._discard_promises(not train)

        self._alloc_promise(policy, train, epoch)
        # Check if remote path has been collected.
        ready_promises = wait(self.promise_list[train], timeout=0)
        for promise in ready_promises:
            rollout = promise.recv()
            path_epoch, _ = self.promise_info[promise]
            self._free_promise(promise)
            # Throw away eval paths from previous epochs
            if path_epoch != epoch and train == False:
                continue
            self._alloc_promise(policy, train, epoch)
            return rollout
        return None

    def update_worker_envs(self, update):
        self.env_update = update

    def shutdown(self):
        for worker in self._parallel_envs:
            worker.terminate()

    def _alloc_promise(self, policy, train, epoch):
        if len(self.free_envs) == 0 or \
           len(self.promise_list[train]) >= self.worker_limits[train]:
            return
        policy_params = policy.get_param_values_np()

        free_env = self.free_envs.pop()
        free_env.send((self.env_update, (policy_params, train,)))
        self.promise_info[free_env] = (epoch, train)
        self.promise_list[train].append(free_env)
        return free_env

    def _free_promise(self, env_id):
        _, train = self.promise_info[env_id]
        assert env_id not in self.free_envs
        if wait([env_id], timeout=0):
            env_id.recv()
        self.free_envs.add(env_id)
        del self.promise_info[env_id]
        self.promise_list[train].remove(env_id)

    def _discard_promises(self, train_type):
        ready_promises = wait(self.promise_list[train_type], timeout=0)
        for promise in ready_promises:
            self._free_promise(promise)


    def _worker_loop(pipe, *worker_env_args, **worker_env_kwargs):
        env = RemoteRolloutEnv.WorkerEnv(*worker_env_args, **worker_env_kwargs)
        while True:
            wait([pipe])
            env_update, rollout_args = pipe.recv()
            if env_update is not None:
                env._env.update_env(**env_update)
            rollout = env.rollout(*rollout_args)
            pipe.send(rollout)

    class WorkerEnv:
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
            self.train_rollout_function = cloudpickle.loads(train_rollout_function)
            self.eval_rollout_function = cloudpickle.loads(eval_rollout_function)

        def rollout(self, policy_params, use_exploration_strategy):
            if use_exploration_strategy:
                self._exploration_policy.set_param_values_np(policy_params)
                policy = self._exploration_policy
                rollout_function = self.train_rollout_function
                env_utils.mode(self._env, 'train')
            else:
                self._policy.set_param_values_np(policy_params)
                policy = self._policy
                rollout_function = self.eval_rollout_function
                env_utils.mode(self._env, 'eval')

            rollout = rollout_function(self._env, policy, self._max_path_length)
            if 'full_observations' in rollout:
                rollout['observations'] = rollout['full_observations'][:-1]
            return rollout
