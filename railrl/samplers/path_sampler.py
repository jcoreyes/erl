import os
import pickle

import numpy as np

from rllab.misc import logger


class MultitaskPathSampler(object):
    def __init__(
            self,
            env,
            exploration_policy,
            exploration_strategy,
            pool,
            min_num_steps_to_collect=1000,
            max_path_length=None,
            render=False,
    ):
        self.env = env
        self.exploration_policy = exploration_policy
        self.exploration_strategy = exploration_strategy
        self.min_num_steps_to_collect = min_num_steps_to_collect
        self.pool = pool
        if max_path_length is None:
            max_path_length = np.inf
        self.max_path_length = max_path_length
        self.render = render

    def collect_data(self):
        obs = self.env.reset()
        n_steps_total = 0
        path_length = 0
        while True:
            action, agent_info = (
                self.exploration_strategy.get_action(
                    n_steps_total,
                    obs,
                    self.exploration_policy,
                )
            )

            next_ob, raw_reward, terminal, env_info = (
                self.env.step(action)
            )
            if self.render:
                self.env.render()
            n_steps_total += 1
            path_length += 1
            reward = raw_reward

            self.pool.add_sample(
                obs,
                action,
                reward,
                terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
            if terminal or path_length >= self.max_path_length:
                if n_steps_total >= self.min_num_steps_to_collect:
                    break
                self.pool.terminate_episode(
                    next_ob,
                    terminal,
                    agent_info=agent_info,
                    env_info=env_info,
                )
                obs = self.reset_env()
                path_length = 0
                logger.log(
                    "Episode Done. # steps done = {}/{} ({:2.2f} %)".format(
                        n_steps_total,
                        self.min_num_steps_to_collect,
                        100 * n_steps_total / self.min_num_steps_to_collect,
                    )
                )
            else:
                obs = next_ob

    def save_pool(self):
        # train_file = os.path.join(dir_name, 'train.pkl')
        # validation_file = os.path.join(dir_name, 'validation.pkl')
        out_dir = logger.get_snapshot_dir()
        filename = os.path.join(out_dir, 'data.pkl')
        with open(filename, 'wb') as handle:
            pickle.dump(self.pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved to {}".format(filename))

    def reset_env(self):
        self.exploration_strategy.reset()
        self.exploration_policy.reset()
        return self.env.reset()