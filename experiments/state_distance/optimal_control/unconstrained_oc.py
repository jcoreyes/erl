import argparse
import os
import random

import joblib

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.policies import (
    UnconstrainedOcWithGoalConditionedModel,
    UnconstrainedOcWithImplicitModel)
from rlkit.state_distance.old.networks import \
    VectorizedGoalStructuredUniversalQfunction
from state_distance.rollout_util import multitask_rollout
from rlkit.core import logger


def experiment(variant):
    num_rollouts = variant['num_rollouts']
    data = joblib.load(variant['qf_path'])
    qf = data['qf']
    env = data['env']
    qf_policy = data['policy']
    if ptu.gpu_enabled():
        qf.to(ptu.device)
        qf_policy.to(ptu.device)
    if isinstance(qf, VectorizedGoalStructuredUniversalQfunction):
        policy = UnconstrainedOcWithImplicitModel(
            qf,
            env,
            qf_policy,
            **variant['policy_params']
        )
    else:
        policy = UnconstrainedOcWithGoalConditionedModel(
            qf,
            env,
            qf_policy,
            **variant['policy_params']
        )
    paths = []
    for _ in range(num_rollouts):
        goal = env.sample_goal_for_rollout()
        print("goal", goal)
        path = multitask_rollout(
            env,
            policy,
            goal,
            **variant['rollout_params']
        )
        paths.append(path)
    env.log_diagnostics(paths)
    logger.dump_tabular(with_timestamp=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file with a QF')
    parser.add_argument('--nrolls', type=int, default=10,
                        help='Number of rollouts to do.')
    parser.add_argument('--H', type=int, default=100, help='Horizon.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--discount', type=float, help='Discount Factor',
                        default=10)
    parser.add_argument('--nsamples', type=int, default=10000,
                        help='Number of samples for optimization')
    parser.add_argument('--dt', help='decrement tau', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    parser.add_argument('--ndc', help='not (decrement and cycle tau)',
                        action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    run_mode = 'none'
    use_gpu = True

    discount = args.discount

    variant = dict(
        num_rollouts=args.nrolls,
        rollout_params=dict(
            max_path_length=args.H,
            animated=not args.hide,
            discount=discount,
            cycle_tau=args.cycle or not args.ndc,
            decrement_discount=args.dt or not args.ndc,
        ),
        policy_params=dict(
            sample_size=args.nsamples,
        ),
        qf_path=os.path.abspath(args.file),
    )
    if run_mode == 'none':
        for exp_id in range(n_seeds):
            seed = random.randint(0, 999999)
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                seed=seed,
                mode=mode,
                variant=variant,
                exp_id=exp_id,
                use_gpu=use_gpu,
            )
