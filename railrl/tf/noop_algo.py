from railrl.algos.bptt_ddpg import BpttDDPG


class NoOpIfyAlgo(object):
    """
    Override the _do_training method of a class to do nothing.

    Usage:
    ```
    algo_class = NoOpIfyAlgo(DDPG)
    algo = algo_classj(param)
    ```
    """
    def __init__(self, algo_class):
        self.algo_class = algo_class

    def _do_training(self, *arg, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        instance = self.algo_class(*args, **kwargs)
        instance._do_training = self._do_training
        return instance


class NoOpBpttDDPG(BpttDDPG):
    def _do_training(self, *arg, **kwargs):
        pass
