"""
:author: Vitchyr Pong
"""

import tensorflow as tf

from railrl.algos.ddpg import DDPG
from railrl.data_management.subtraj_replay_buffer import (
    SubtrajReplayBuffer
)
from railrl.misc.rllab_util import split_flat_product_space_into_components_n
from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.pythonplusplus import map_recursive
from railrl.qfunctions.nn_qfunction import NNQFunction

TARGET_PREFIX = "target_"


class BpttDDPG(DDPG):
    """
    The ICML idea: this does DDPG updates, but also does BPTT assuming you
    have a recurrent policy.
    """

    def __init__(
            self,
            env,
            exploration_strategy,
            policy: RnnCellPolicy,
            qf: NNQFunction,
            num_bptt_unrolls=1,
            env_obs_dim=None,
            replay_pool_size=10000,
            replay_buffer_class=SubtrajReplayBuffer,
            **kwargs
    ):
        """
        :param env: Environment
        :param exploration_strategy: ExplorationStrategy
        :param policy: Policy that is Serializable
        :param qf: QFunctions that is Serializable
        :param qf_learning_rate: Learning rate of the qf
        :param policy_learning_rate: Learning rate of the _policy
        :param qf_weight_decay: How much to decay the weights for Q
        :return:
        """
        self._num_bptt_unrolls = num_bptt_unrolls
        self._env_obs_dim = env_obs_dim

        self._rnn_cell_scope = policy.rnn_cell_scope
        self._rnn_cell = policy.rnn_cell

        self._replay_buffer_class = replay_buffer_class
        super().__init__(
            env,
            exploration_strategy,
            policy,
            qf,
            replay_pool=replay_buffer_class(
                replay_pool_size,
                env,
                num_bptt_unrolls,
            ),
            **kwargs)

    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        rnn_inputs = tf.unpack(self._rnn_inputs_ph, axis=1)
        self._rnn_init_state_ph = self.policy.get_init_state_placeholder()

        self._rnn_cell_scope.reuse_variables()
        with tf.variable_scope(self._rnn_cell_scope):
            self._rnn_outputs, self._rnn_final_state = tf.nn.rnn(
                self._rnn_cell,
                rnn_inputs,
                initial_state=self._rnn_init_state_ph,
                dtype=tf.float32,
                scope=self._rnn_cell_scope,
            )
        self._final_rnn_output = self._rnn_outputs[-1]
        self._final_rnn_action = (
            self._final_rnn_output,
            self._rnn_final_state,
        )
        # TODO(vitchyr): consider taking the sum of the outputs rather than
        # only the last output.

        # To compute the surrogate loss function for the qf, it must take
        # as input the output of the _policy. See Equation (6) of "Deterministic
        # Policy Gradient Algorithms" ICML 2014.
        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=self._final_rnn_action
        )
        self.policy_surrogate_loss = - tf.reduce_mean(
            self.qf_with_action_input.output)
        self.train_policy_op = tf.train.AdamOptimizer(
            self.policy_learning_rate).minimize(
            self.policy_surrogate_loss,
            var_list=self.policy.get_params_internal())

    def _sample_minibatch(self):
        return self.pool.random_subtrajectories(self.batch_size)

    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs,
                          **kwargs):
        actions = self._split_flat_actions(actions)
        obs = self._split_flat_obs(obs)
        next_obs = self._split_flat_obs(next_obs)

        # rewards and terminals both have shape [batch_size x sub_traj_length],
        # but they really just need to be [batch_size x 1]. Right now we only
        # care about the reward/terminal at the very end since we're only
        # computing the rewards for the last time step.
        qf_terminals = terminals[:, -1]
        qf_rewards = rewards[:, -1]
        # For obs/actions, we only care about the last time step for the critic.
        qf_obs = self._get_time_step(obs, t=-1)
        qf_actions = self._get_time_step(actions, t=-1)
        qf_next_obs = self._get_time_step(next_obs, t=-1)
        feed = self._qf_feed_dict(qf_rewards,
                                  qf_terminals,
                                  qf_obs,
                                  qf_actions,
                                  qf_next_obs,
                                  **kwargs)

        policy_feed = self._policy_feed_dict(obs, **kwargs)
        feed.update(policy_feed)
        return feed

    @staticmethod
    def _get_time_step(action_or_obs, t):
        """
        Squeeze time out by only taking the one time step.

        :param action_or_obs: tuple of Tensors or Tensor of shape [batch size x
        traj length x dim]
        :param t: The time index to slice out.
        :return: return tuple of Tensors or Tensor of shape [batch size x dim]
        """
        return map_recursive(lambda x: x[:, t, :], action_or_obs)

    def _policy_feed_dict(self, obs, **kwargs):
        """
        :param obs: See output of `self._split_flat_action`.
        :return: Feed dictionary for policy training TensorFlow ops.
        """
        last_obs = self._get_time_step(obs, -1)
        first_obs = self._get_time_step(obs, 0)
        env_obs, _ = obs
        initial_memory_obs = first_obs[1]
        return {
            self.qf_with_action_input.observation_input: last_obs,
            self._rnn_inputs_ph: env_obs,
            self._rnn_init_state_ph: initial_memory_obs,
        }

    def _update_feed_dict_from_path(self, paths):
        eval_pool = self._replay_buffer_class(
            len(paths) * self.max_path_length,
            self.env,
            self._num_bptt_unrolls,
        )
        for path in paths:
            eval_pool.add_trajectory(path)

        batch = eval_pool.get_all_valid_subtrajectories()
        return self._update_feed_dict(
            rewards=batch['rewards'],
            terminals=batch['terminals'],
            obs=batch['observations'],
            actions=batch['actions'],
            next_obs=batch['next_observations'],
        )
