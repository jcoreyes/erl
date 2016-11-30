import abc


class OptimizableQFunction(object):
    """
    A Q-function that implicitly has a _policy.
    """
    @abc.abstractmethod
    def get_implicit_policy(self):
        return