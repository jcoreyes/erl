from collections import OrderedDict

from rlkit.core.timer import timer

from rlkit.core import logger
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.misc import eval_util
from rlkit.samplers.data_collector.path_collector import PathCollector
from rlkit.core.rl_algorithm import BaseRLAlgorithm

def linear_schedule(start, end, current):
    return float(current) / (end - start)

class BatchRLAlgorithmModEnv(BaseRLAlgorithm):
    def __init__(
            self,
            batch_size,
            max_path_length,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            mod_env_epoch_schedule,
            env_class,
            env_mod_params,
            num_train_loops_per_epoch=1,
            min_num_steps_before_training=0,
            *args,
            **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training

        self.mod_env_epoch_schedule = mod_env_epoch_schedule
        self.env_class = env_class
        self.env_mod_params = env_mod_params

    def _train(self):
        done = (self.epoch == self.num_epochs)
        if done:
            return OrderedDict(), done

        if self.epoch == 0 and self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        timer.start_timer('evaluation sampling')
        if self.epoch % self._eval_epoch_freq == 0:
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
        timer.stop_timer('evaluation sampling')

        if not self._eval_only:
            for _ in range(self.num_train_loops_per_epoch):
                timer.start_timer('exploration sampling', unique=False)

                current = max(1, self.epoch / (self.mod_env_epoch_schedule * self.num_epochs))
                new_env_mod_parms = dict()
                for k, v in self.env_mod_params.items():
                    new_env_mod_parms[k] = 1.0 * current + (1.0 - current) * v
                self.expl_data_collector._env = self.env_class(new_env_mod_parms)

                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                timer.stop_timer('exploration sampling')

                timer.start_timer('replay buffer data storing', unique=False)
                self.replay_buffer.add_paths(new_expl_paths)
                timer.stop_timer('replay buffer data storing')

                timer.start_timer('training', unique=False)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(self.batch_size)
                    self.trainer.train(train_data)
                timer.stop_timer('training')
        log_stats = self._get_diagnostics()
        return log_stats, False
