from rllab.sampler import parallel_sampler


class SimplePathSampler(object):
    """
    Sample things in another thread by serializing the policy and environment.
    Only one thread is used.
    """
    def __init__(self, env, policy, max_samples, max_path_length):
        self.env = env
        self.policy = policy
        self.max_samples = max_samples
        self.max_path_length = max_path_length

    def start_worker(self):
        parallel_sampler.populate_task(self.env, self.policy)

    def shutdown_worker(self):
        parallel_sampler.terminate_task()

    def obtain_samples(self):
        cur_params = self.policy.get_param_values()
        return parallel_sampler.sample_paths(
            policy_params=cur_params,
            max_samples=self.max_samples,
            max_path_length=self.max_path_length,
        )