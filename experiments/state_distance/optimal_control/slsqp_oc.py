import argparse
import random
import joblib
import os

from railrl.algos.state_distance.state_distance_q_learning import \
    multitask_rollout
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.state_distance import (
    ArgmaxQFPolicy,
    PseudoModelBasedPolicy,
    ConstrainedOptimizationOCPolicy,
)
from rllab.misc import logger
import railrl.torch.pytorch_util as ptu


def experiment(variant):
    num_rollouts = variant['num_rollouts']
    H = variant['H']
    render = variant['render']
    data = joblib.load(variant['qf_path'])
    qf = data['qf']
    env = data['env']
    if ptu.gpu_enabled():
        qf.cuda()
    policy = variant['policy_class'](
        qf,
        env,
        **variant['policy_params']
    )
    paths = []
    for _ in range(num_rollouts):
        goal = env.sample_goal_state_for_rollout()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file with a QF')
    parser.add_argument('--nrolls', type=int, default=5,
                        help='Number of rollouts to do.')
    parser.add_argument('--H', type=int, default=100, help='Horizon.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    n_seeds = 1
    mode = "local"
    exp_prefix = "dev"
    run_mode = 'none'
    use_gpu = True

    variant = dict(
        num_rollouts=args.nrolls,
        H=args.H,
        render=not args.hide,
        policy_class=ConstrainedOptimizationOCPolicy,
        policy_params=dict(
            solver_params=dict(
                disp=args.verbose,
                maxiter=10,
            )
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
