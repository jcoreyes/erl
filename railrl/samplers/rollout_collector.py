from railrl.samplers.rollout_functions import rollout


class MdpPathCollector(object):
    def __init__(
            self,
            env,
            policy,
            max_path_length,
            num_paths,
    ):
        self._env = env
        self._policy = policy
        self._max_path_length = max_path_length
        self._num_paths = num_paths

    def get(self):
        return [
            rollout(
                self._env, self._policy, max_path_length=self._max_path_length
            )
            for _ in range(self._num_paths)
        ]
