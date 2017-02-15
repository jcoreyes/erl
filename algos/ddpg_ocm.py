"""
:author: Vitchyr Pong
"""
from collections import OrderedDict

import numpy as np

from railrl.algos.ddpg import DDPG
from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from rllab.misc import logger
from rllab.misc import special
from rllab.misc.overrides import overrides

TARGET_PREFIX = "target_"


class DdpgOcm(DDPG):
    """
    Deep Deterministic Policy Gradient for one character memory task.
    """

    @overrides
    def _get_training_ops(self, epoch=None):
        ops = [
            self.train_qf_op,
            self.update_target_qf_op,
        ]
        if epoch > 50:
            ops += [
                self.train_policy_op,
                self.update_target_policy_op,
            ]
        if self._batch_norm:
            ops += self.qf.batch_norm_update_stats_op
            ops += self.policy.batch_norm_update_stats_op
        return ops

    @overrides
    def evaluate(self, epoch, es_path_returns):
        logger.log("Collecting samples for evaluation")
        paths = self._sample_paths(epoch)
        self.log_diagnostics(paths)
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)

        # Compute statistics
        (
            policy_loss,
            qf_loss,
            policy_output,
            target_policy_output,
            qf_output,
            target_qf_outputs,
            ys,
        ) = self.sess.run(
            [
                self.policy_surrogate_loss,
                self.qf_loss,
                self.policy.output,
                self.target_policy.output,
                self.qf.output,
                self.target_qf.output,
                self.ys,
            ],
            feed_dict=feed_dict)
        discounted_returns = [
            special.discount_return(path["rewards"], self.discount)
            for path in paths]
        returns = [sum(path["rewards"]) for path in paths]
        rewards = np.hstack([path["rewards"] for path in paths])

        # Log statistics
        last_statistics = OrderedDict([
            ('Epoch', epoch),
            ('AverageReturn', np.mean(returns)),
            ('PolicySurrogateLoss', policy_loss),
            ('QfLoss', qf_loss),
        ])
        last_statistics.update(create_stats_ordered_dict('Ys', ys))
        last_statistics.update(create_stats_ordered_dict('PolicyOutput',
                                                         policy_output))
        last_statistics.update(create_stats_ordered_dict('TargetPolicyOutput',
                                                         target_policy_output))
        last_statistics.update(create_stats_ordered_dict('QfOutput', qf_output))
        last_statistics.update(create_stats_ordered_dict('TargetQfOutput',
                                                         target_qf_outputs))
        last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        last_statistics.update(create_stats_ordered_dict('Returns', returns))
        last_statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                         discounted_returns))
        if len(es_path_returns) > 0:
            last_statistics.update(create_stats_ordered_dict('TrainingReturns',
                                                             es_path_returns))

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

    def get_epoch_snapshot(self, epoch):
        return dict(
            env=self.training_env,
            epoch=epoch,
            policy=self.policy,
            es=self.exploration_strategy,
            qf=self.qf,
        )
