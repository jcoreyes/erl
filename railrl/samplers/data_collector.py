import abc
from collections import deque, OrderedDict

from railrl.samplers.rollout_functions import rollout


class PathCollector(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def collect_new_paths(self, num_paths):
        pass

    @abc.abstractmethod
    def get_epoch_paths(self):
        pass

    def end_epoch(self):
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}


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

        self._num_steps_total = 0
        self._num_paths_total = 0

    def collect_new_paths(self, num_paths):
        paths = [
            rollout(
                self._env, self._policy, max_path_length=self._max_path_length
            )
            for _ in range(num_paths)
        ]
        self._num_paths_total += len(paths)
        self._num_steps_total += sum(
            map(lambda path: len(path['actions']), paths)
        )
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        return OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
        ])

    def get_snapshot(self):
        return dict(
            env=self._env,
            policy=self._policy,
        )
