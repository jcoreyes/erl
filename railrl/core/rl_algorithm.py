import abc
from collections import OrderedDict

from railrl.core.timer import timer

from railrl.core import logger
from railrl.core.logging import append_log
from railrl.misc import eval_util
from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.samplers.data_collector import DataCollector


def _get_epoch_timings():
    times_itrs = timer.get_times()
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: DataCollector,
            evaluation_data_collector: DataCollector,
            replay_buffer: ReplayBuffer,
            num_epochs,
            exploration_get_diagnostic_functions=None,
            evaluation_get_diagnostic_functions=None,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0
        self.post_train_funcs = []
        self.post_epoch_funcs = []
        self.epoch = self._start_epoch
        self.num_epochs = num_epochs
        if exploration_get_diagnostic_functions is None:
            exploration_get_diagnostic_functions = [
                eval_util.get_generic_path_information,
            ]
            if hasattr(self.expl_env, 'get_diagnostics'):
                exploration_get_diagnostic_functions.append(
                    self.expl_env.get_diagnostics)
        if evaluation_get_diagnostic_functions is None:
            evaluation_get_diagnostic_functions = [
                eval_util.get_generic_path_information,
            ]
            if hasattr(self.eval_env, 'get_diagnostics'):
                evaluation_get_diagnostic_functions.append(
                    self.eval_env.get_diagnostics)
        self._eval_get_diag_fns = evaluation_get_diagnostic_functions
        self._expl_get_diag_fns = exploration_get_diagnostic_functions

    def train(self):
        timer.return_global_times = True
        for _ in range(self.num_epochs):
            self._begin_epoch()
            logger.save_itr_params(self.epoch, self._get_snapshot())
            timer.stamp('saving')
            log_dict, _ = self._train()
            logger.record_dict(log_dict)
            logger.dump_tabular(with_prefix=True, with_timestamp=False)
            self._end_epoch()
        logger.save_itr_params(self.epoch, self._get_snapshot())

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _begin_epoch(self):
        timer.reset()

    def _end_epoch(self):
        for post_train_func in self.post_train_funcs:
            post_train_func(self, self.epoch)

        self.expl_data_collector.end_epoch(self.epoch)
        self.eval_data_collector.end_epoch(self.epoch)
        self.replay_buffer.end_epoch(self.epoch)
        self.trainer.end_epoch(self.epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, self.epoch)
        self.epoch += 1

    def _get_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.replay_buffer.get_snapshot().items():
            snapshot['replay_buffer/' + k] = v
        return snapshot

    def _get_diagnostics(self):
        algo_log = OrderedDict()
        append_log(algo_log, self.replay_buffer.get_diagnostics(),
                   prefix='replay_buffer/')
        append_log(algo_log, self.trainer.get_diagnostics(), prefix='trainer/')
        # Exploration
        append_log(algo_log, self.expl_data_collector.get_diagnostics(),
                   prefix='exploration/')
        expl_paths = self.expl_data_collector.get_epoch_paths()
        for fn in self._expl_get_diag_fns:
            append_log(algo_log, fn(expl_paths), prefix='exploration/')
        # Eval
        append_log(algo_log, self.eval_data_collector.get_diagnostics(),
                   prefix='evaluation/')
        eval_paths = self.eval_data_collector.get_epoch_paths()
        for fn in self._eval_get_diag_fns:
            append_log(algo_log, fn(eval_paths), prefix='evaluation/')

        timer.stamp('logging')
        append_log(algo_log, _get_epoch_timings())
        algo_log['epoch'] = self.epoch
        return algo_log

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
