import abc
from collections import deque

from railrl.samplers.rollout_functions import rollout


class PathCollector(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(self, num_paths):
        pass

    @abc.abstractmethod
    def get_diagnostics(self):
        pass

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass

    @abc.abstractmethod
    def end_epoch(self):
        pass

    @abc.abstractmethod
    def update_snapshot(self):
        pass


class MdpPathCollector(object):
    def __init__(
            self,
            env,
            policy,
            max_path_length,
            max_num_epoch_paths_saved=32,
    ):
        self._env = env
        self._policy = policy
        self._max_path_length = max_path_length
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def collect_new_paths(self, num_paths):
        paths = [
            rollout(
                self._env, self._policy, max_path_length=self._max_path_length
            )
            for _ in range(num_paths)
        ]
        self._epoch_paths.extend(paths)
        return paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def update_snapshot(self, snapshot):
        return snapshot
