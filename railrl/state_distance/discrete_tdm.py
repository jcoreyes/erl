import numpy as np
from railrl.misc.ml_util import ConstantSchedule
from railrl.torch.algos.dqn import DQN
from railrl.torch.algos.util import np_to_pytorch_batch


class DiscreteTDM(DQN):
    def __init__(
            self,
            env,
            qf,
            max_tau=10,
            epoch_max_tau_schedule=None,
            sample_train_goals_from='replay_buffer',
            sample_rollout_goals_from='environment',
            **kwargs
    ):
        """

        :param env:
        :param qf:
        :param epoch_max_tau_schedule: A schedule for the maximum planning
        horizon tau.
        :param kwargs:
        """
        super().__init__(env, qf, **kwargs)

        if epoch_max_tau_schedule is None:
            epoch_max_tau_schedule = ConstantSchedule(max_tau)

        self.max_tau = max_tau
        self.epoch_max_tau_schedule = epoch_max_tau_schedule
        self.sample_train_goals_from = sample_train_goals_from
        self.sample_rollout_goals_from = sample_rollout_goals_from
        self.goal = None

    def _start_epoch(self, epoch):
        self.max_tau = self.epoch_max_tau_schedule.get_value(epoch)
        super()._start_epoch(epoch)

    def get_batch(self, training=True):
        if self.replay_buffer_is_split:
            replay_buffer = self.replay_buffer.get_replay_buffer(training)
        else:
            replay_buffer = self.replay_buffer
        batch = replay_buffer.random_batch(self.batch_size)

        """
        Update the goal states/rewards
        """
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        goals = self.sample_goals_for_training()
        rewards = self.compute_rewards_np(
            obs,
            actions,
            next_obs,
            goals,
        )
        batch['rewards'] = rewards

        """
        Update the tau
        """
        num_steps_left = np.random.randint(
            0, self.max_tau + 1, (self.batch_size, 1)
        )
        batch['num_steps_left'] = num_steps_left

        return np_to_pytorch_batch(batch)

    def compute_rewards_np(self, obs, actions, next_obs, goals):
        return self.env.compute_rewards(
            obs,
            actions,
            next_obs,
            goals,
        )

    @property
    def train_buffer(self):
        if self.replay_buffer_is_split:
            return self.replay_buffer.get_replay_buffer(trainig=True)
        else:
            return self.replay_buffer

    def sample_goals_for_training(self):
        if self.sample_train_goals_from == 'environment':
            return self.env.sample_goals(self.batch_size)
        elif self.sample_train_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(self.batch_size)
            obs = batch['observations']
            return self.env.convert_obs_to_goals(obs)
        elif self.sample_train_goals_from == 'her':
            raise Exception("Take samples from replay buffer.")
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_train_goals_from
            ))
    def sample_goal_for_rollout(self):
        if self.sample_rollout_goals_from == 'environment':
            return self.env.sample_goal_for_rollout()
        elif self.sample_rollout_goals_from == 'replay_buffer':
            batch = self.train_buffer.random_batch(1)
            obs = batch['observations']
            goal_state = self.env.convert_obs_to_goal_states(obs)[0]
            return self.env.modify_goal_for_rollout(goal_state)
        else:
            raise Exception("Invalid `sample_goals_from`: {}".format(
                self.sample_rollout_goals_from
            ))

    def offline_evaluate(self, epoch):
        raise NotImplementedError()

    def _start_new_rollout(self):
        self.exploration_policy.reset()
        self.goal = self.sample_goal_for_rollout()
        # self.training_env.set_goal(self.goal_state)
        # self.exploration_policy.set_goal(self.goal_state)
        # self.exploration_policy.set_discount(self.discount)
        return self.training_env.reset()

    def _handle_step(
            self,
            observation,
            action,
            reward,
            next_observation,
            terminal,
            agent_info,
            env_info,
    ):
        self._current_path_builder.add_all(
            observations=observation,
            actions=action,
            rewards=reward,
            next_observations=next_observation,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
            goals=self.goal,
            # taus=self._rollout_discount,
        )

        self.replay_buffer.add_sample(
            observation=observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            agent_info=agent_info,
            env_info=env_info,
            goal=self.goal,
        )
        #
        # if self.cycle_taus_for_rollout:
        #     self._rollout_discount -= 1
        #     if self._rollout_discount < 0:
        #         self._rollout_discount = self.discount
        #     self.exploration_policy.set_discount(self._rollout_discount)

