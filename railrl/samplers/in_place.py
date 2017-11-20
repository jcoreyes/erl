from railrl.samplers.util import rollout


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_samples, max_path_length):
        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.num_rollouts = max_samples // self.max_path_length
        assert self.num_rollouts > 0, "Need max_samples >= max_path_length"

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self):
        return [
            rollout(self.env, self.policy, max_path_length=self.max_path_length)
            for _ in range(self.num_rollouts)
        ]