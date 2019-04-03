from collections import OrderedDict

import gtimer as gt

from railrl.core import logger
import railrl.core.logging as logging
from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.misc import eval_util
from railrl.samplers.data_collector import PathCollector
from railrl.core.logging import add_log


def _get_epoch_timings(epoch):
    times_itrs = gt.get_times().stamps.itrs
    train_time = times_itrs['training'][-1]
    expl_sampling_time = times_itrs['exploration sampling'][-1]
    data_storing_time = times_itrs['data storing'][-1]
    save_time = times_itrs['saving'][-1]
    eval_sampling_time = times_itrs['evaluation sampling'][-1] if epoch > 0 else 0
    epoch_time = train_time + expl_sampling_time + eval_sampling_time
    total_time = gt.get_times().total

    return OrderedDict([
        ('time/data storing (s)', data_storing_time),
        ('time/training (s)', train_time),
        ('time/evaluation sampling (s)', eval_sampling_time),
        ('time/exploration sampling (s)', expl_sampling_time),
        ('time/saving (s)', save_time),
        ('time/epoch (s)', epoch_time),
        ('time/total train (s)', total_time),
    ])


class BatchRLAlgorithm(object):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: PathCollector,
            evaluation_data_collector: PathCollector,
            data_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        self.trainer = trainer
        self.expl_env = exploration_env
        self.eval_env = evaluation_env
        self.expl_data_collector = exploration_data_collector
        self.eval_data_collector = evaluation_data_collector
        self.data_buffer = data_buffer
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self._start_epoch = 0
        # self.epoch_list = iter(gt.timed_for(
                # range(self._start_epoch, self.num_epochs),
                # save_itrs=True,))
        self.epoch_list = iter(range(self._start_epoch, self.num_epochs))
        self.epoch = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state['epoch_list'] = None
        return state

    def  __setstate__(self, state):
        start_epoch = 0
        if state['epoch'] != 0:
            start_epoch = state['epoch']
        self.__dict__ = state
        # self.epoch_list = iter(gt.timed_for(
                # range(start_epoch, self.num_epochs),
                # save_itrs=True,))
        self.epoch_list = iter(range(start_epoch, self.num_epochs))

    def train(self, start_epoch=0):
        self._start_epoch = start_epoch
        self.epoch = self._start_epoch
        for epoch in self.epoch_list:
            self._train()

    def _train(self):
        self.epoch = next(self.epoch_list)
        if self.epoch == 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
            )
            self.data_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
        )
        # gt.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            new_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_expl_steps_per_train_loop,
            )
            # gt.stamp('exploration sampling', unique=False)

            self.data_buffer.add_paths(new_expl_paths)
            # gt.stamp('data storing', unique=False)

            for _ in range(self.num_trains_per_train_loop):
                train_data = self.data_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            # gt.stamp('training', unique=False)
        self._save_snapshot()
        algo_logs = self.get_diagnostics()
        self._end_epoch()
        done = False
        if self.epoch == self.num_epochs - 1:
            done = True
        return algo_logs, done

    def _save_snapshot(self):
        snapshot = {}
        for k, v in self.trainer.get_snapshot().items():
            snapshot['trainer/' + k] = v
        for k, v in self.expl_data_collector.get_snapshot().items():
            snapshot['exploration/' + k] = v
        for k, v in self.eval_data_collector.get_snapshot().items():
            snapshot['evaluation/' + k] = v
        for k, v in self.data_buffer.get_snapshot().items():
            snapshot['buffer/' + k] = v
        logger.save_itr_params(self.epoch, snapshot)
        # gt.stamp('saving')

    def get_diagnostics(self):
        logger.log("Epoch {} finished".format(self.epoch), with_timestamp=True)
        algorithm_logs = {}
        add_log(algorithm_logs,
                self.data_buffer.get_diagnostics(),
                prefix='buffer/')
        add_log(algorithm_logs,
                self.trainer.get_diagnostics(),
                prefix='trainer/')
        add_log(algorithm_logs,
                self.expl_data_collector.get_diagnostics(),
                prefix='exploration/')
        expl_paths = self.expl_data_collector.get_epoch_paths()
        if hasattr(self.expl_env, 'get_diagnostics'):
            add_log(algorithm_logs,
                    self.expl_env.get_diagnostics(expl_paths),
                    prefix='exploration/')
        add_log(algorithm_logs,
                eval_util.get_generic_path_information(expl_paths),
                prefix="exploration/")
        add_log(algorithm_logs,
                self.eval_data_collector.get_diagnostics(),
                prefix='evaluation/')
        eval_paths = self.eval_data_collector.get_epoch_paths()
        if hasattr(self.eval_env, 'get_diagnostics'):
            add_log(algorithm_logs,
                    self.eval_env.get_diagnostics(eval_paths),
                    prefix='evaluation/')
        add_log(algorithm_logs,
                eval_util.get_generic_path_information(eval_paths),
                prefix="evaluation/")
        """
        Misc
        """
        # algorithm_logs.update(_get_epoch_timings(self.epoch))
        algorithm_logs['epoch'] = self.epoch
        # logger.record_tabular('Epoch', epoch)
        # logger.dump_tabular(with_prefix=False, with_timestamp=False)
        return algorithm_logs

    def _end_epoch(self):
        self.expl_data_collector.end_epoch(self.epoch)
        self.eval_data_collector.end_epoch(self.epoch)
        self.data_buffer.end_epoch(self.epoch)
        self.trainer.end_epoch(self.epoch)


