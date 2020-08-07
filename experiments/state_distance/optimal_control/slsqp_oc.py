import argparse
import os
import random

import joblib

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.policies import (
    ArgmaxQFPolicy,
    PseudoModelBasedPolicy,
    StateOnlySdqBasedSqpOcPolicy)
from state_distance.rollout_util import multitask_rollout
from rlkit.core import logger


def experiment(variant):
    num_rollouts = variant['num_rollouts']
    H = variant['H']
    render = variant['render']
    data = joblib.load(variant['qf_path'])
    qf = data['qf']
    env = data['env']
    qf_policy = data['policy']
    if ptu.gpu_enabled():
        qf.to(ptu.device)
        qf_policy.to(ptu.device)
    policy_class = variant['policy_class']
    if policy_class == StateOnlySdqBasedSqpOcPolicy:
        policy = policy_class(
            qf,
            env,
            qf_policy,
            **variant['policy_params']
        )
    else:
        policy = policy_class(
            qf,
            env,
            **variant['policy_params']
        )
    paths = []
    for _ in range(num_rollouts):
        goal = env.sample_goal_for_rollout()
        path = multitask_rollout(
            env,
            policy,
            goal,
            discount=variant['discount'],
            max_path_length=H,
            animated=render,
        )
        paths.append(path)
    env.log_diagnostics(paths)
    logger.dump_tabular(with_timestamp=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file with a QF')
    parser.add_argument('--nrolls', type=int, default=5,
                        help='Number of rollouts to do.')
    parser.add_argument('--H', type=int, default=100, help='Horizon.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--planh', type=int, default=5,
                        help='Planning horizon.')
    parser.add_argument('--maxi', type=int, default=5,
                        help='Max SLSQP steps per env step.')
    parser.add_argument('--ftol', type=float, default=1e-2,
                        help='Tolerance for constraint optimizer')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    run_mode = 'none'
    use_gpu = True

    discount = 0
    if args.discount is not None:
        print("WARNING: you are overriding the discount factor. Right now "
              "only discount = 0 really makes sense.")
        discount = args.discount

    variant = dict(
        num_rollouts=args.nrolls,
        H=args.H,
        render=not args.hide,
        policy_class=StateOnlySdqBasedSqpOcPolicy,
        policy_params=dict(
            solver_params=dict(
                disp=args.verbose,
                maxiter=args.maxi,
                ftol=args.ftol,
                iprint=1,
            ),
            planning_horizon=args.planh,
        ),
        qf_path=os.path.abspath(args.file),
        discount=discount,
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
    elif run_mode == 'custom':
        for (policy_class, policy_params) in [
            (
                    PseudoModelBasedPolicy,
                    dict(
                        sample_size=1,
                        num_gradient_steps=100,
                    )
            ),
            (
                    PseudoModelBasedPolicy,
                    dict(
                        sample_size=100,
                        num_gradient_steps=1,
                    )
            ),
            (
                    ArgmaxQFPolicy,
                    dict(
                        sample_size=1,
                        num_gradient_steps=100,
                    )
            ),
            (
                    ArgmaxQFPolicy,
                    dict(
                        sample_size=100,
                        num_gradient_steps=1,
                    )
            ),
        ]:
            variant['policy_class'] = policy_class
            variant['policy_params'] = policy_params
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
