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

        # OCM-specific statistics
        import ipdb
        ipdb.set_trace()
        target_onehots = []
        predictions = []
        for path in paths:
            env_obs = [env_obs for env_obs, _, in path["observations"]]
            target_onehots.append(np.argmax(env_obs[0, :]))
            env_actions = [env_action for env_action, _ in path["actions"]]
            predictions.append(env_actions)
        final_predictions = predictions[-1]  # batch_size X dim
        nonfinal_predictions = predictions[:-1]  # list of batch_size X dim
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
