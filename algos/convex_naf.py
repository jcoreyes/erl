"""
:author: Vitchyr Pong
"""
from collections import OrderedDict

import numpy as np

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.misc.rllab_util import split_paths
from railrl.algos.naf import NAF
from rllab.misc import logger, special
from rllab.misc.overrides import overrides


class ConvexNAFAlgorithm(NAF):
    """
    Continuous Q-learning with Normalized Advantage Function, but where the
    advantage function is just some convex function w.r.t. its input
    """

    @overrides
    def _init_tensorflow_ops(self):
        super()._init_tensorflow_ops()
        self._af = self.qf.advantage_function
        self._qf_update_weights_ops = self.qf.update_weights_ops
        if self._qf_update_weights_ops is not None:
            self.sess.run(self._qf_update_weights_ops)

    @overrides
    def _get_training_ops(self):
        ops = [[
            self.train_qf_op,
            self.update_target_vf_op,
        ]]
        if self._qf_update_weights_ops is not None:
            ops.append(self._qf_update_weights_ops)
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
        policy_output = [self.policy.get_action(o)[0] for o in obs]
        (
            qf_loss,
            qf_output,
            target_vf_output,
            ys,
        ) = self.sess.run(
            [
                self.qf_loss,
                self.qf.output,
                self.target_vf.output,
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
            ('QfLoss', qf_loss),
        ])
        last_statistics.update(create_stats_ordered_dict('Ys', ys))
        last_statistics.update(create_stats_ordered_dict('QfOutput', qf_output))
        last_statistics.update(create_stats_ordered_dict('TargetVfOutput',
                                                         target_vf_output))
        last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        last_statistics.update(create_stats_ordered_dict('Returns', returns))
        last_statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                         discounted_returns))
        last_statistics.update(create_stats_ordered_dict('PolicyOutput',
                                                         policy_output))
        if len(es_path_returns) > 0:
            last_statistics.update(create_stats_ordered_dict('TrainingReturns',
                                                             es_path_returns))
        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return self.last_statistics

    # @overrides
    # def _update_feed_dict(self, rewards, terminals, obs, actions, next_obs):
    #     feed_dict = super()._update_feed_dict(rewards, terminals, obs,
    #                                           actions, next_obs)
    #     current_policy_actions = np.vstack(
    #         [self.policy.get_action(o)[0] for o in obs]
    #     )
    #     feed_dict[self.qf.policy_output_placeholder] = current_policy_actions
    #     return feed_dict
