import argparse
import json

import joblib
from pathlib import Path

import railrl.torch.pytorch_util as ptu
from railrl.misc.eval_util import get_generic_path_information
from railrl.state_distance.rollout_util import multitask_rollout
from railrl.core import logger
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mtau', type=float,
                        help='Max tau value')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--dt', help='decrement tau', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    parser.add_argument('--ndc', help='not (decrement and cycle tau)',
                        action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    if args.mtau is None:
        variant_path = Path(args.file).parents[0] / 'variant.json'
        variant = json.load(variant_path.open())
        try:
            max_tau = variant['sac_tdm_kwargs']['tdm_kwargs']['max_tau']
            print("Max tau read from variant: {}".format(max_tau))
        except KeyError:
            print("Defaulting max tau to 10.")
            max_tau = 10
    else:
        max_tau = args.mtau

    env = data['env']
    num_samples = 1000
    resolution = 10
    if 'policy' in data:
        policy = data['policy']
    else:
        policy = data['exploration_policy']
    policy.train(False)
    if args.pause:
        import ipdb; ipdb.set_trace()

    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.cuda()

    while True:
        paths = []
        for _ in range(args.nrolls):
            goal = env.sample_goal_for_rollout()
            print("goal", goal)
            path = multitask_rollout(
                env,
                policy,
                goal,
                init_tau=max_tau,
                max_path_length=args.H,
                animated=not args.hide,
                cycle_tau=args.cycle or not args.ndc,
                decrement_tau=args.dt or not args.ndc,
                get_action_kwargs={'deterministic': True},
            )
            paths.append(path)
        env.log_diagnostics(paths)
        for key, value in get_generic_path_information(paths).items():
            logger.record_tabular(key, value)
        logger.dump_tabular()
