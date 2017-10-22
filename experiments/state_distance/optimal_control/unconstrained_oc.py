import argparse
import random
import joblib
import os

from railrl.algos.state_distance.state_distance_q_learning import \
    multitask_rollout
from railrl.launchers.launcher_util import run_experiment
from railrl.policies.state_distance import (
    UnconstrainedOcWithGoalConditionedModel,
)
from rllab.misc import logger
import railrl.torch.pytorch_util as ptu

def experiment(variant):
    num_rollouts = variant['num_rollouts']
    data = joblib.load(variant['qf_path'])
    qf = data['qf']
    env = data['env']
    qf_policy = data['policy']
    if ptu.gpu_enabled():
        qf.cuda()
        qf_policy.cuda()
    policy = UnconstrainedOcWithGoalConditionedModel(
        qf,
        env,
        qf_policy,
        **variant['policy_params']
    )
    paths = []
    for _ in range(num_rollouts):
        goal = env.sample_goal_state_for_rollout()
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
    parser.add_argument('--nrolls', type=int, default=5,
                        help='Number of rollouts to do.')
    parser.add_argument('--H', type=int, default=100, help='Horizon.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--discount', type=float, help='Discount Factor')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of samples for optimization')
    parser.add_argument('--dt', help='decrement tau', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    parser.add_argument('--dc', help='decrement and cycle tau',
                        action='store_true')
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
        rollout_params=dict(
            max_path_length=args.H,
            animated=not args.hide,
            discount=discount,
            cycle_tau=args.cycle or args.dc,
            decrement_discount=args.dt or args.dc,
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
