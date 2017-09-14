import time
from railrl.misc.rllab_util import (
    save_extra_data_to_snapshot_dir,
    get_table_key_set,
)
from railrl.torch.naf import NAF
from rllab.misc import logger


class ParallelNAF(NAF):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.num_gradient_steps = 0

    def train(self, start_epoch=0):
        self.training_mode(True)
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        old_table_keys = None

        n_steps_total = 0
        n_steps_current_epoch = 0

        epoch_start_time = time.time()
        exploration_paths = []
        epoch = 0
        logger.push_prefix('Epoch #%d | ' % epoch)
        while n_steps_total <= self.num_epochs * self.num_steps_per_epoch:
            path = self.training_env.rollout(
                self.exploration_policy,
                use_exploration_strategy=True,
            )
            if path is not None:
                path_length = len(path['observations'])
                n_steps_total += path_length
                n_steps_current_epoch += path_length
                exploration_paths.append(path)

                for (
                        reward,
                        terminal,
                        action,
                        obs,
                        agent_info,
                        env_info
                ) in zip(
                    path["rewards"].reshape(-1, 1),
                    path["terminals"].reshape(-1, 1),
                    path["actions"],
                    path["observations"],
                    path["agent_infos"],
                    path["env_infos"],
                ):
                    self.replay_buffer.add_sample(
                        obs,
                        action,
                        reward,
                        terminal,
                        agent_info=agent_info,
                        env_info=env_info,
                    )
                self.replay_buffer.terminate_episode(
                    path["final_observation"],
                    path["terminals"][-1],
                    agent_info=path["agent_infos"][-1],
                    env_info=path["env_infos"][-1],
                )
                self.handle_rollout_ending(n_steps_total)

            if self._can_train():
                self._do_training(n_steps_total=n_steps_total)

            if n_steps_current_epoch >= self.num_steps_per_epoch:
                if self._can_evaluate(exploration_paths):
                    start_time = time.time()
                    logger.record_tabular(
                        "Number of gradient steps",
                        self.num_gradient_steps,
                    )
                    self.evaluate(epoch, exploration_paths)
                    params = self.get_epoch_snapshot(epoch)
                    logger.save_itr_params(epoch, params)
                    save_extra_data_to_snapshot_dir(
                        self.get_extra_data_to_save(epoch),
                    )
                    table_keys = get_table_key_set(logger)
                    if old_table_keys is not None:
                        assert table_keys == old_table_keys, (
                            "Table keys cannot change from iteration to iteration."
                        )
                    old_table_keys = table_keys
                    logger.dump_tabular(with_prefix=False, with_timestamp=False)
                    logger.log("Eval Time: {0}".format(time.time() - start_time))

                epoch_duration = time.time() - epoch_start_time
                logger.log("Epoch duration: {0}".format(epoch_duration))
                logger.pop_prefix()

                self.discount = self.epoch_discount_schedule.get_value(epoch)
                epoch_start_time = time.time()
                exploration_paths = []
                epoch += 1
                n_steps_current_epoch = 0
                logger.push_prefix('Epoch #%d | ' % epoch)

    def _do_training(self, n_steps_total):
        self.num_gradient_steps += 1
        super()._do_training(n_steps_total)

    def _sample_eval_paths(self, epoch):
        """
        Do extra training while sampling paths for evaluation.
        :param epoch:
        :return:
        """
        n_steps_total = 0
        paths = []
        while n_steps_total <= self.num_steps_per_eval:
            path = self.training_env.rollout(
                self.exploration_policy,
                use_exploration_strategy=False,
            )
            if path is not None:
                path_length = len(path['observations'])
                n_steps_total += path_length
                paths.append(path)

            if self._can_train():
                self._do_training(n_steps_total=n_steps_total)
        return paths
