"""
:author: Vitchyr Pong
"""
from collections import OrderedDict
import numpy as np
import tensorflow as tf

from algos.naf import NAF
from misc.data_processing import create_stats_ordered_dict
from misc.rllab_util import split_paths
from rllab.misc.overrides import overrides
from rllab.misc import logger, special


class ConvexNAFAlgorithm(NAF):
    """
    Continuous Q-learning with Normalized Advantage Function, but where the
    advantage function is just some convex function w.r.t. its input
    """

    @overrides
    def _init_tensorflow_ops(self):
        super()._init_tensorflow_ops()
        self.af = self.qf.get_implicit_advantage_function()
        self.clip_weight_ops = [v.assign(tf.maximum(v, 0)) for v in
                                self.af.get_action_params()]

    @overrides
    def _get_training_ops(self):
        return [
            self.train_qf_op,
            self.update_target_vf_op,
        ] + self.clip_weight_ops

    @overrides
    def evaluate(self, epoch, es_path_returns):
        logger.log("Collecting samples for evaluation")
        paths = self.eval_sampler.obtain_samples(
            itr=epoch,
            batch_size=self.n_eval_samples,
        )
        self.env.log_diagnostics(paths)
        rewards, terminals, obs, actions, next_obs = split_paths(paths)
        feed_dict = self._update_feed_dict(rewards, terminals, obs, actions,
                                           next_obs)

        # Compute statistics
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
            ('CriticLoss', qf_loss),
        ])
        last_statistics.update(create_stats_ordered_dict('Ys', ys))
        last_statistics.update(create_stats_ordered_dict('QfOutput', qf_output))
        last_statistics.update(create_stats_ordered_dict('TargetVfOutput',
                                                         target_vf_output))
        last_statistics.update(create_stats_ordered_dict('Rewards', rewards))
        last_statistics.update(create_stats_ordered_dict('Returns', returns))
        last_statistics.update(create_stats_ordered_dict('DiscountedReturns',
                                                         discounted_returns))
        policy_output = [self.policy.get_action(o)[0] for o in obs]
        last_statistics.update(create_stats_ordered_dict('PolicyOutput',
                                                         policy_output))
        if len(es_path_returns) > 0:
            last_statistics.update(create_stats_ordered_dict('TrainingReturns',
                                                             es_path_returns))
        for key, value in last_statistics.items():
            logger.record_tabular(key, value)

        return self.last_statistics
