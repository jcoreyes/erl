"""
:author: Vitchyr Pong
"""
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from railrl.algos.ddpg import DDPG
from railrl.data_management.episode_replay_buffer import EpisodeReplayBuffer
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths, \
    split_flat_product_space_into_components_n
from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.pythonplusplus import map_recursive
from railrl.qfunctions.nn_qfunction import NNQFunction
from rllab.misc import logger
from rllab.misc import special

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

        super().__init__(
            env,
            exploration_strategy,
            policy,
            qf,
            replay_pool=EpisodeReplayBuffer(
                self._num_bptt_unrolls,
                env,
            ),
            **kwargs)

    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        self._rnn_init_state_ph = self.policy.create_init_state_placeholder()

        rnn_inputs = tf.unpack(self._rnn_inputs_ph, axis=1)

        self._rnn_cell_scope.reuse_variables()
        with tf.variable_scope(self._rnn_cell_scope):
            self._rnn_outputs, self._rnn_final_state = tf.nn.rnn(
                self._rnn_cell,
                rnn_inputs,
                # initial_state=self._rnn_init_state_ph,
                initial_state=tf.split(1, 2, self._rnn_init_state_ph),
                dtype=tf.float32,
                scope=self._rnn_cell_scope,
            )
        self._final_rnn_output = self._rnn_outputs[-1]
        self._final_rnn_action = (
            self._final_rnn_output,
            tf.concat(1, self._rnn_final_state),
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

    def _split_flat_obs(self, obs):
        """
        :param obs: [batch_size X num_bbpt_unroll X (env_obs_dim + memory_dim)
        :return: Tuple with
         - [batch_size X num_bbpt_unroll X env_obs_dim
         - [batch_size X num_bbpt_unroll X memory_dim
        """
        return split_flat_product_space_into_components_n(
            self.env.spec.observation_space,
            obs,
        )

    def _split_flat_actions(self, actions):
        """

        :param actions: [batch_size x num_bptt_unroll x (env_action_dim +
        memory_dim)
        :return: Tuple with
         - [batch_size X num_bbpt_unroll X env_action_dim
         - [batch_size X num_bbpt_unroll X memory_dim
        """
        return split_flat_product_space_into_components_n(
            self.env.spec.action_space,
            actions,
        )

    def _do_training(self, epoch=None):
        minibatch = self.pool.random_subtrajectories(
            self.batch_size,
            self._num_bptt_unrolls,
        )
        sampled_obs = minibatch['observations']
        sampled_terminals = minibatch['terminals']
        sampled_actions = minibatch['actions']
        sampled_rewards = minibatch['rewards']
        sampled_next_obs = minibatch['next_observations']

        feed_dict = self._update_feed_dict(sampled_rewards,
                                           sampled_terminals,
                                           sampled_obs,
                                           sampled_actions,
                                           sampled_next_obs)
        ops = self._get_training_ops(epoch=epoch)
        if isinstance(ops[0], list):
            for op in ops:
                self.sess.run(op, feed_dict=feed_dict)
        else:
            self.sess.run(ops, feed_dict=feed_dict)

    def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        actions = self._split_flat_actions(actions)
        obs = self._split_flat_obs(obs)
        next_obs = self._split_flat_obs(next_obs)

        # rewards and terminals both have shape [batch_size x sub_traj_length],
        # but they really just need to be [batch_size x 1]. Right now we only
        # care about the reward/terminal at the very end since we're only
        # computing the rewards for the last time step.
        qf_terminals = terminals[:, -1:]
        qf_rewards = rewards[:, -1:]
        # For obs/actions, we only care about the last time step for the critic.
        qf_obs = self._get_time_step(obs, t=-1)
        qf_actions = self._get_time_step(actions, t=-1)
        qf_next_obs = self._get_time_step(next_obs, t=-1)
        feed = self._qf_feed_dict(qf_rewards,
                                  qf_terminals,
                                  qf_obs,
                                  qf_actions,
                                  qf_next_obs)

        policy_feed = self._policy_feed_dict(obs)
        feed.update(policy_feed)
        return feed

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.rewards_placeholder: rewards,
            self.terminals_placeholder: terminals,
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.target_qf.observation_input: next_obs,
            self.target_policy.observation_input: next_obs,
        }

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

    def _policy_feed_dict(self, obs):
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

    def evaluate(self, epoch, es_path_returns):
        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        self.log_diagnostics(paths)
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        # feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
        #                                    next_obs)
        # feed_dict = self._qf_feed_dict(rewards, terminals, obs, actions, next_obs)

        last_statistics = OrderedDict()

        # # Compute statistics
        # (
        #     policy_loss,
        #     qf_loss,
        #     policy_output,
        #     target_policy_output,
        #     qf_output,
        #     target_qf_outputs,
        #     ys,
        # ) = self.sess.run(
        #     [
        #         self.policy_surrogate_loss,
        #         self.qf_loss,
        #         self.policy.output,
        #         self.target_policy.output,
        #         self.qf.output,
        #         self.target_qf.output,
        #         self.ys,
        #     ],
        #     feed_dict=feed_dict)
        # discounted_returns = [
        #     special.discount_return(path["rewards"], self.discount)
        #     for path in paths]
        # returns = [sum(path["rewards"]) for path in paths]
        # rewards = np.hstack([path["rewards"] for path in paths])
        #
        # # Log statistics
        # last_statistics = OrderedDict([
        #     ('Epoch', epoch),
        #     ('AverageReturn', np.mean(returns)),
        #     ('PolicySurrogateLoss', policy_loss),
        #     ('QfLoss', qf_loss),
        # ])
        # last_statistics.update(create_stats_ordered_dict('Ys', ys))
        # last_statistics.update(create_stats_ordered_dict('PolicyOutput',
        #                                                  policy_output))
        # last_statistics.update(create_stats_ordered_dict('TargetPolicyOutput',
        #                                                  target_policy_output))
        # last_statistics.update(create_stats_ordered_dict('QfOutput', qf_output))
        # last_statistics.update(create_stats_ordered_dict('TargetQfOutput',
        #                                                  target_qf_outputs))
        # last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        # last_statistics.update(create_stats_ordered_dict('Returns', returns))
        # last_statistics.update(create_stats_ordered_dict('DiscountedReturns',
        #                                                  discounted_returns))
        # if len(es_path_returns) > 0:
        #     last_statistics.update(create_stats_ordered_dict('TrainingReturns',
        #                                                      es_path_returns))

        """
        OCM-specific statistics
        """
        target_onehots = []
        for path in paths:
            first_observation = path["observations"][0]
            first_env_obs, _ = self._split_flat_obs(first_observation)
            target_onehots.append(first_env_obs)

        final_predictions = []  # each element has shape (dim)
        nonfinal_predictions = []  # each element has shape (seq_length-1, dim)
        for path in paths:
            env_actions = np.array([self._split_flat_actions(a)[0] for a in
                                    path["actions"]])
            final_predictions.append(env_actions[-1])
            nonfinal_predictions.append(env_actions[:-1])
        nonfinal_predictions_sequence_dimension_flattened = np.vstack(
            nonfinal_predictions
        )  # shape = N X dim

        """
        Calculate statistics
        """

        nonfinal_prob_zero = [softmax[0] for softmax in
                              nonfinal_predictions_sequence_dimension_flattened]
        final_probs_correct = []
        for final_prediction, target_onehot in zip(final_predictions,
                                                   target_onehots):
            correct_pred_idx = np.argmax(target_onehot)
            final_probs_correct.append(final_prediction[correct_pred_idx])
        final_prob_zero = [softmax[0] for softmax in final_predictions]

        last_statistics.update(create_stats_ordered_dict(
            'Final P(correct)',
            final_probs_correct))
        last_statistics.update(create_stats_ordered_dict(
            'Non-final P(zero)',
            nonfinal_prob_zero))
        last_statistics.update(create_stats_ordered_dict(
            'Final P(zero)',
            final_prob_zero))

        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return last_statistics
