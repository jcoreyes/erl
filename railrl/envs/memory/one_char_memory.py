import numpy as np
from sklearn.metrics import log_loss
from random import randint

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.np_util import np_print_options, softmax
from railrl.misc.rllab_util import split_flat_product_space_into_components_n
from collections import OrderedDict
from railrl.pythonplusplus import clip_magnitude
from rllab.envs.base import Env
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import logger
from railrl.envs.supervised_learning_env import RecurrentSupervisedLearningEnv
from cached_property import cached_property
import tensorflow as tf


class OneCharMemory(Env, RecurrentSupervisedLearningEnv):
    """
    A simple env whose output is a value `X` the first time step, followed by a
    fixed number of zeros.

    The goal of the agent is to output zero for all time steps, and then
    output `X` in the last time step.

    Both the actions and observations are represented as probability vectors.
    There are `n` different values that `X` can take on (excluding 0),
    so the probability vector's dimension is n+1.

    The reward is the negative cross-entropy loss between the target one-hot
    vector and the probability vector outputted by the agent. Furthermore, the
    reward for the last time step is multiplied by `reward_for_remember`.
    """

    def __init__(
            self,
            n=4,
            num_steps=10,
            reward_for_remembering=1,
            max_reward_magnitude=1,
            softmax_action=False,
    ):
        """
        :param n: Number of different values that could be returned
        :param num_steps: How many steps the agent needs to remember.
        :param reward_for_remembering: The reward bonus for remembering the
        number. This number is added to the usual reward if the correct
        number has the maximum probability.
        :param max_reward_magnitude: Clip the reward magnitude to this value.
        :param softmax_action: If true, put the action through a softmax.
        """
        assert max_reward_magnitude >= reward_for_remembering
        super().__init__()
        self.num_steps = num_steps
        self.n = n
        self._onehot_size = n + 1
        self._action_space = Box(
            np.zeros(self._onehot_size),
            np.ones(self._onehot_size)
        )
        self._observation_space = self._action_space
        self._t = 1
        self._reward_for_remembering = reward_for_remembering
        self._max_reward_magnitude = max_reward_magnitude
        self._softmax_action = softmax_action

        self._target_number = None

        # For rendering
        self._last_reward = None
        self._last_action = None
        self._last_t = None

    def step(self, action):
        if self._softmax_action:
            action = softmax(action)
        # Reset gives the first observation, so only return 0 in step.
        observation = self._get_next_observation(0)
        done = self._t == self.num_steps
        info = self._get_info_dict()
        reward = self._compute_reward(done, action)

        self._last_t = self._t
        self._t += 1

        self._last_reward = reward
        self._last_action = action
        return observation, reward, done, info

    def _get_info_dict(self):
        return {
            'target_number': self._target_number,
            'time': self._t - 1,
        }

    def _compute_reward(self, done, action):
        try:
            if done:
                reward = -log_loss(self._get_target_onehot(), action)
                if np.argmax(action) == self._target_number:
                    reward += self._reward_for_remembering
            else:
                reward = -log_loss(self.zero, action)
            # if reward == -np.inf:
            #     reward = -self._max_reward_magnitude
            # if reward == np.inf or np.isnan(reward):
            #     reward = self._max_reward_magnitude
        except ValueError as e:
            raise e
            # reward = -self._max_reward_magnitude
        reward = clip_magnitude(reward, self._max_reward_magnitude)
        return reward

    @cached_property
    def zero(self):
        z = np.zeros(self._onehot_size)
        z[0] = 1
        return z

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    def reset(self):
        self._target_number = randint(1, self.n)
        self._t = 1
        first_observation = self._get_next_observation(self._target_number)
        return first_observation

    def _get_next_observation(self, observation_int):
        return special.to_onehot(observation_int, self._onehot_size)

    def _get_target_onehot(self):
        return special.to_onehot(self._target_number, self._onehot_size)

    @property
    def observation_space(self):
        return self._observation_space

    def get_batch(self, batch_size):
        targets = np.random.randint(
            low=1,
            high=self.n+1,
            size=batch_size,
        )
        onehot_targets = special.to_onehot_n(targets, self.feature_dim)
        X = np.zeros((batch_size, self.sequence_length, self.feature_dim))
        X[:, :, 0] = 1  # make the target 0
        X[:, 0, :] = onehot_targets
        Y = np.zeros((batch_size, self.sequence_length, self.target_dim))
        Y[:, :, 0] = 1  # make the target 0
        Y[:, -1, :] = onehot_targets
        return X, Y

    @property
    def feature_dim(self):
        return self.n + 1

    @property
    def target_dim(self):
        return self.n + 1

    @property
    def sequence_length(self):
        return self.horizon

    @overrides
    def render(self):
        logger.push_prefix("OneCharMemory(n={0})\t".format(self._target_number))
        if self._last_action is None:
            logger.log("No action taken.")
        else:
            if self._last_t == 1:
                logger.log("--- New Episode ---")
            logger.push_prefix("t={0}\t".format(self._last_t))
            with np_print_options(precision=4, suppress=False):
                logger.log("Action: {0}".format(
                    self._last_action,
                ))
            logger.log("Reward: {0}".format(
                self._last_reward,
            ))
            logger.pop_prefix()
        logger.pop_prefix()

    def log_diagnostics(self, paths):
        target_onehots = []
        for path in paths:
            first_observation = path["observations"][0]
            target_onehots.append(first_observation)

        final_predictions = []  # each element has shape (dim)
        nonfinal_predictions = []  # each element has shape (seq_length-1, dim)
        for path in paths:
            actions = path["actions"]
            if self._softmax_action:
                actions = softmax(actions, axis=-1)
            final_predictions.append(actions[-1])
            nonfinal_predictions.append(actions[:-1])
        nonfinal_predictions_sequence_dimension_flattened = np.vstack(
            nonfinal_predictions
        )  # shape = N X dim
        nonfinal_prob_zero = [softmax[0] for softmax in
                              nonfinal_predictions_sequence_dimension_flattened]
        final_probs_correct = []
        for final_prediction, target_onehot in zip(final_predictions,
                                                   target_onehots):
            correct_pred_idx = np.argmax(target_onehot)
            final_probs_correct.append(final_prediction[correct_pred_idx])
        final_prob_zero = [softmax[0] for softmax in final_predictions]

        last_statistics = OrderedDict()
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

        return final_probs_correct

    def get_tf_loss(self, observations, actions, target_labels):
        """
        Return the supervised-learning loss.
        :param observation: Tensor
        :param action: Tensor
        :return: loss Tensor
        """
        target_labels_float = tf.cast(target_labels, tf.float32)
        cross_entropy = target_labels_float * tf.log(actions)
        return tf.reduce_sum(cross_entropy, axis=1)


class OneCharMemoryEndOnly(OneCharMemory):
    """
    Don't reward or penalize outputs other than the last output. Then,
    only give a 1 or 0.
    """
    def _compute_reward(self, done, action):
        if done:
            if np.argmax(action) == self._target_number:
                return self._reward_for_remembering
            else:
                return - self._reward_for_remembering
        return 0


class OneCharMemoryEndOnlyLogLoss(OneCharMemory):
    """
    Don't reward or penalize outputs other than the last output. Then,
    give the usual reward.
    """
    def _compute_reward(self, done, action):
        if done:
            return super()._compute_reward(done, action)
        return 0


class OneCharMemoryEndOnlyDiscrete(OneCharMemory):
    """
    TODO: finish this class
    A simple env whose output is a value `X` the first time step, followed by a
    fixed number of zeros.

    The goal of the agent is to output zero for all time steps, and then
    output `X` in the last time step.

    Both the actions and observations are represented as one hot vectors.

    The reward is the indicator function of whether or not the policy
    outputted the correct thing.
    """

    def __init__(
            self,
            n=4,
            num_steps=10,
            reward_for_remembering=1,
            max_reward_magnitude=1,
    ):
        """
        :param n: Number of different values that could be returned
        :param num_steps: How many steps the agent needs to remember.
        :param reward_for_remembering: The reward bonus for remembering the
        number. This number is added to the usual reward if the correct
        number has the maximum probability.
        :param max_reward_magnitude: Clip the reward magnitude to this value.
        """
        assert max_reward_magnitude >= reward_for_remembering
        self.num_steps = num_steps
        self.n = n
        self._onehot_size = n + 1
        self._action_space = Discrete(self._onehot_size)
        self._observation_space = self._action_space
        self._t = 1

        self._target_number = None

        # For rendering
        self._last_reward = None
        self._last_action = None
        self._last_t = None

    def _compute_reward(self, done, action):
        if done:
            reward = int(self._get_target_onehot() == action)
        else:
            reward = 0
        return reward

    @cached_property
    def zero(self):
        z = np.zeros(self._onehot_size)
        z[0] = 1
        return z

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self.num_steps

    def reset(self):
        self._target_number = randint(1, self.n)
        self._t = 1
        first_observation = self._get_next_observation(self._target_number)
        return first_observation

    def _get_next_observation(self, observation_int):
        return special.to_onehot(observation_int, self._onehot_size)

    def _get_target_onehot(self):
        return special.to_onehot(self._target_number, self._onehot_size)

    @property
    def observation_space(self):
        return self._observation_space

    def get_batch(self, batch_size):
        targets = np.random.randint(
            low=1,
            high=self.n+1,
            size=batch_size,
        )
        onehot_targets = special.to_onehot_n(targets, self.feature_dim)
        X = np.zeros((batch_size, self.sequence_length, self.feature_dim))
        X[:, :, 0] = 1  # make the target 0
        X[:, 0, :] = onehot_targets
        Y = np.zeros((batch_size, self.sequence_length, self.target_dim))
        Y[:, :, 0] = 1  # make the target 0
        Y[:, -1, :] = onehot_targets
        return X, Y

    @property
    def feature_dim(self):
        return self.n + 1

    @property
    def target_dim(self):
        return self.n + 1

    @property
    def sequence_length(self):
        return self.horizon

    @overrides
    def render(self):
        logger.push_prefix("OneCharMemory(n={0})\t".format(self._target_number))
        if self._last_action is None:
            logger.log("No action taken.")
        else:
            if self._last_t == 1:
                logger.log("--- New Episode ---")
            logger.push_prefix("t={0}\t".format(self._last_t))
            with np_print_options(precision=4, suppress=False):
                logger.log("Action: {0}".format(
                    self._last_action,
                ))
            logger.log("Reward: {0}".format(
                self._last_reward,
            ))
            logger.pop_prefix()
        logger.pop_prefix()

    def log_diagnostics(self, paths):
        target_onehots = []
        for path in paths:
            first_observation = path["observations"][0]
            target_onehots.append(first_observation)

        final_predictions = []  # each element has shape (dim)
        nonfinal_predictions = []  # each element has shape (seq_length-1, dim)
        for path in paths:
            actions = path["actions"]
            final_predictions.append(actions[-1])
            nonfinal_predictions.append(actions[:-1])
        nonfinal_predictions_sequence_dimension_flattened = np.vstack(
            nonfinal_predictions
        )  # shape = N X dim
        nonfinal_prob_zero = [softmax[0] for softmax in
                              nonfinal_predictions_sequence_dimension_flattened]
        final_probs_correct = []
        for final_prediction, target_onehot in zip(final_predictions,
                                                   target_onehots):
            correct_pred_idx = np.argmax(target_onehot)
            final_probs_correct.append(final_prediction[correct_pred_idx])
        final_prob_zero = [softmax[0] for softmax in final_predictions]

        last_statistics = OrderedDict()
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
