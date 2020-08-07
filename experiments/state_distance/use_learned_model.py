"""
Use a learned dynamics model to solve a task.
"""
import argparse
import os
import random

import joblib

import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import run_experiment
from rlkit.state_distance.old.model_based_policies import (
    MultistepModelBasedPolicy,
    SQPModelBasedPolicy,
)
from rlkit.state_distance.old.networks import ModelExtractor
from state_distance.rollout_util import multitask_rollout
from rlkit.core import logger


def experiment(variant):
    num_rollouts = variant['num_rollouts']
    H = variant['H']
    render = variant['render']
    data = joblib.load(variant['qf_path'])
    policy_params = variant['policy_params']
    if 'model' in data:
        model = data['model']
    else:
        qf = data['qf']
        model = ModelExtractor(qf)
        policy_params['model_learns_deltas'] = False
    env = data['env']
    if ptu.gpu_enabled():
        model.to(ptu.device)
    policy = variant['policy_class'](
        model,
        env,
        **policy_params
    )
    paths = []
    for _ in range(num_rollouts):
        goal = env.sample_goal_for_rollout()
        path = multitask_rollout(
            env,
            policy,
            goal,
            discount=0,
            max_path_length=H,
            animated=render,
        )
        paths.append(path)
    env.log_diagnostics(paths)
    logger.dump_tabular(with_timestamp=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--planh', type=int, default=5,
                        help='Planning horizon')
    parser.add_argument('--nrolls', type=int, default=5,
                        help='Number of rollouts to do.')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of sample for sampled-based optimizer')
    parser.add_argument('--maxi', type=int, default=5,
                        help='Max number of iterations for sqp')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--nosqp', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev-use-learned-model"
    run_mode = 'none'
    use_gpu = True

    if args.nosqp:
        policy_class = MultistepModelBasedPolicy
        policy_params = dict(
            model_learns_deltas=True,
            sample_size=args.nsamples,
            planning_horizon=args.planh,
        )
    else:
        policy_class = SQPModelBasedPolicy
        policy_params = dict(
            model_learns_deltas=True,
            solver_params=dict(
                disp=args.verbose,
                maxiter=args.maxi,
                ftol=1e-2,
                iprint=1,
            ),
            planning_horizon=args.planh,
        )

    variant = dict(
        num_rollouts=args.nrolls,
        H=args.H,
        render=not args.hide,
        qf_path=os.path.abspath(args.file),
        policy_class=policy_class,
        policy_params=policy_params,
    )

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
