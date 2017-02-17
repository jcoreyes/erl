"""
:author: Vitchyr Pong
"""
from collections import OrderedDict
from typing import List

import numpy as np
import tensorflow as tf

from railrl.core.neuralnet import NeuralNetwork
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from railrl.algos.ddpg import DDPG
from railrl.policies.memory.rnn_cell_policy import RnnCellPolicy
from railrl.policies.nn_policy import NNPolicy
from railrl.qfunctions.nn_qfunction import NNQFunction
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.spaces.product import Product

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
            qf_learning_rate=1e-3,
            policy_learning_rate=1e-4,
            qf_weight_decay=0.,
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
        self.qf = qf
        self.qf_learning_rate = qf_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.qf_weight_decay = qf_weight_decay
        self._num_bptt_unrolls = num_bptt_unrolls
        self._env_obs_dim = env_obs_dim

        self._rnn_cell_scope = policy.rnn_cell_scope
        self._rnn_cell = policy.rnn_cell

        super().__init__(env, policy, exploration_strategy, **kwargs)

    def _init_policy_ops(self):
        self._rnn_inputs_ph = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_time_inputs',
        )
        self._rnn_init_state = tf.placeholder(
            tf.float32,
            [None, self._num_bptt_unrolls, self._env_obs_dim],
            name='rnn_init_state',

        )

        rnn_inputs = tf.unpack(self._rnn_inputs_ph, axis=1)

        self._rnn_cell_scope.reuse_variables()
        with tf.variable_scope(self._rnn_cell_scope):
            self._rnn_outputs, self._rnn_final_state = tf.nn.rnn(
                self._rnn_cell,
                rnn_inputs,
                initial_state=self._rnn_init_state,
                dtype=tf.float32,
                scope=self._rnn_cell_scope,
            )
        self._final_rnn_output = self._rnn_outputs[-1]

        # To compute the surrogate loss function for the qf, it must take
        # as input the output of the _policy. See Equation (6) of "Deterministic
        # Policy Gradient Algorithms" ICML 2014.
        self.qf_with_action_input = self.qf.get_weight_tied_copy(
            action_input=self._final_rnn_output
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
        # TODO
        return self.env.spec.observation_space.split_flat_into_components_n(
            obs
        )

    def _split_flat_actions(self, actions):
        """

        :param actions: [batch_size x num_bptt_unroll x (env_action_dim +
        memory_dim)
        :return: Tuple with
         - [batch_size X num_bbpt_unroll X env_action_dim
         - [batch_size X num_bbpt_unroll X memory_dim
        """
        # TODO
        return self.env.spec.action_space.split_flat_into_components_n(
            actions
        )

    def _qf_feed_dict(self, rewards, terminals, obs, actions, next_obs):
        return {
            self.rewards_placeholder: np.expand_dims(rewards, axis=1),
            self.terminals_placeholder: np.expand_dims(terminals, axis=1),
            self.qf.observation_input: obs,
            self.qf.action_input: actions,
            self.target_qf.observation_input: next_obs,
            self.target_policy.observation_input: next_obs,
        }

    def _policy_feed_dict(self, obs):
        return {
            self.qf_with_action_input.observation_input: obs,
            self.policy.observation_input: obs,
        }
