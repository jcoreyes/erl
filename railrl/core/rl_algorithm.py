import abc
from collections import OrderedDict

import gtimer as gt

from railrl.core import logger
from railrl.misc import eval_util
from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.samplers.data_collector import BaseCollector
from railrl.core.logging import append_log

def _get_epoch_timings():
    times_itrs = gt.get_times().stamps.itrs
    times = OrderedDict()
    epoch_time = 0
    for key in sorted(times_itrs):
        time = times_itrs[key][-1]
        epoch_time += time
        times['time/{} (s)'.format(key)] = time
    times['time/epoch (s)'] = epoch_time
    times['time/total (s)'] = gt.get_times().total
    return times


class BaseRLAlgorithm(object, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: BaseCollector,
            evaluation_data_collector: BaseCollector,
            replay_buffer: ReplayBuffer,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.replay_buffer = replay_buffer
        self._start_epoch = 0
        self.post_epoch_funcs = []
        self.epoch = self._start_epoch

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            self._train()

    def _train(self):
        """
        Train model.
        """
        raise NotImplementedError('_train must implemented by inherited class')

    def _end_epoch(self):
        self.expl_data_collector.end_epoch(self.epoch)
        self.eval_data_collector.end_epoch(self.epoch)
        self.replay_buffer.end_epoch(self.epoch)
        self.trainer.end_epoch(self.epoch)

        for post_epoch_func in self.post_epoch_funcs:
            post_epoch_func(self, self.epoch)
        self.epoch += 1

    def _get_diagnostics(self):
        algo_log = {}
        algo_log['epoch'] = self.epoch
        append_log(algo_log, self.replay_buffer.get_diagnostics(),
                   prefix='replay_buffer/')
        append_log(algo_log, self.trainer.get_diagnostics(), prefix='trainer/')
        append_log(algo_log, self.expl_data_collector.get_diagnostics(),
                   prefix='exploration/')

        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            append_log(algo_log, self.expl_env.get_diagnostics(expl_paths),
                       prefix='exploration/')
        append_log(algo_log, eval_util.get_generic_path_information(expl_paths),
                   prefix="exploration/")

        append_log(algo_log, self.eval_data_collector.get_diagnostics(),
                   prefix='evaluation/')
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            append_log(algo_log, self.eval_env.get_diagnostics(eval_paths),
                       prefix='evaluation/')
        append_log(algo_log,
                   eval_util.get_generic_path_information(eval_paths),
                   prefix="evaluation/")
        return algo_log
        # gt.stamp('logging')
        # logger.record_dict(_get_epoch_timings())
        # logger.record_tabular('Epoch', epoch)
        # logger.dump_tabular(with_prefix=False, with_timestamp=False)

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass
