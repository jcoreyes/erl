"""
Script to compare different optimal control variants for a give qf.
"""

import argparse
from collections import defaultdict

import joblib
import numpy as np

from railrl.algos.state_distance.state_distance_q_learning import \
    multitask_rollout
from railrl.launchers.launcher_util import run_experiment
from railrl.misc.rllab_util import get_logger_table_dict
from railrl.policies.state_distance import (
    SoftOcOneStepRewardPolicy,
    TerminalRewardSampleOCPolicy,
    ArgmaxQFPolicy,
    PseudoModelBasedPolicy,
    SamplePolicyPartialOptimizer,
)
import railrl.misc.hyperparameter as hyp
from railrl.samplers.util import rollout
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


def get_class_params_to_try(policy_class):
    if policy_class == SoftOcOneStepRewardPolicy:
        search_space = {
            'sample_size': [100, 1000],
            'constraint_weight': [1, 10]
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(search_space)
        return sweeper.iterate_hyperparameters()
    if policy_class == ArgmaxQFPolicy:
        return [
            {
                'sample_size': 1,
                'num_gradient_steps': 100,
            },
            {
                'sample_size': 1000,
                'num_gradient_steps': 0,
            },
        ]
    if policy_class == SamplePolicyPartialOptimizer:
        search_space = {
            'sample_size': [100, 1000],
        }
        sweeper = hyp.DeterministicHyperparameterSweeper(search_space)
        return sweeper.iterate_hyperparameters()


def experiment(variant):
    path = variant['path']
    policy_class = variant['policy_class']
    policy_params = variant['policy_params']
    horizon = variant['horizon']
    num_rollouts = variant['num_rollouts']
    discount = variant['discount']
    stat_name = variant['stat_name']

    data = joblib.load(path)
    env = data['env']
    qf = data['qf']
    qf_argmax_policy = data['policy']
    policy = policy_class(
        qf,
        env,
        qf_argmax_policy,
        **policy_params
    )

    paths = []
    for _ in range(num_rollouts):
        goal = env.sample_goal_state_for_rollout()
        path = multitask_rollout(
            env,
            policy,
            goal,
            discount,
            max_path_length=horizon,
            animated=False,
            decrement_discount=False,
        )
        path['goal_states'] = np.repeat(
            np.expand_dims(goal, 0),
            len(path['observations']),
            0,
        )
        paths.append(path)
    env.log_diagnostics(paths)
    results = get_logger_table_dict()
    logger.dump_tabular()
    return results[stat_name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file with a QF')
    args = parser.parse_args()

    exp_prefix = 'sdql-compare-ocs'
    variant = dict(
        path=args.file,
        stat_name='Final Euclidean distance to goal Mean',
        horizon=50,
        num_rollouts=1,
        discount=5,
    )

    policy_to_scores = defaultdict(list)
    exp_id = 0
    for policy_class in [
        SoftOcOneStepRewardPolicy,
        SamplePolicyPartialOptimizer,
        ArgmaxQFPolicy,
    ]:
        variant['policy_class'] = policy_class
        for policy_params in get_class_params_to_try(policy_class):
            variant['policy_params'] = policy_params
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                variant=variant,
                exp_id=exp_id,
            )
            exp_id += 1
            # score = get_score(env, policy, stat_name, horizon, num_rollouts,
            #                   discount)
            # policy_to_scores[policy_class].append(score)
    # import ipdb;
    #
    # ipdb.set_trace()
    # print(policy_to_scores)
