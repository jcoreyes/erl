import abc

from railrl.core.rl_algorithm import BaseRLAlgorithm
from railrl.core.timer import timer
from railrl.data_management.replay_buffer import ReplayBuffer
from railrl.samplers.data_collector import (
    PathCollector,
    StepCollector,
)


class OnlineRLAlgorithm(BaseRLAlgorithm, metaclass=abc.ABCMeta):
    def __init__(
            self,
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector: StepCollector,
            evaluation_data_collector: PathCollector,
            replay_buffer: ReplayBuffer,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
    ):
        super().__init__(
            trainer,
            exploration_env,
            evaluation_env,
            exploration_data_collector,
            evaluation_data_collector,
            replay_buffer,
        )
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        assert self.num_trains_per_train_loop >= self.num_expl_steps_per_train_loop, \
            'Online training presumes num_trains_per_train_loop >= num_expl_steps_per_train_loop'
    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        self.training_mode(False)
        if self.min_num_steps_before_training > 0 && self.epoch == 0:
            self.expl_data_collector.collect_new_steps(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            init_expl_paths = self.expl_data_collector.get_epoch_paths()
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        num_trains_per_expl_step = self.num_trains_per_train_loop // self.num_expl_steps_per_train_loop
        self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True,
        )
        timer.stamp('evaluation sampling')

        for _ in range(self.num_train_loops_per_epoch):
            for _ in range(self.num_expl_steps_per_train_loop):
                self.expl_data_collector.collect_new_steps(
                    self.max_path_length,
                    1,  # num steps
                    discard_incomplete_paths=False,
                )
                timer.stamp('exploration sampling', unique=False)

                self.training_mode(True)
                for _ in range(num_trains_per_expl_step):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                timer.stamp('training', unique=False)
                self.training_mode(False)

        new_expl_paths = self.expl_data_collector.get_epoch_paths()
        self.replay_buffer.add_paths(new_expl_paths)
        timer.stamp('data storing', unique=False)

        log_stats = self._get_diagnostics()
        self._end_epoch(epoch)

        return log_stats, False
