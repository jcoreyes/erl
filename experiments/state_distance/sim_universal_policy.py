import argparse
import json

import joblib
from pathlib import Path

import rlkit.torch.pytorch_util as ptu
from rlkit.misc.eval_util import get_generic_path_information
from rlkit.state_distance.rollout_util import multitask_rollout
from rlkit.core import logger
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
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--mode', type=str, help='env mode')
    parser.add_argument('--silent', action='store_true')
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
            print("Defaulting max tau to 0.")
            max_tau = 0
    else:
        max_tau = args.mtau

    env = data['env']
    num_samples = 1000
    resolution = 10
    max_tau=15
    if 'policy' in data:
        policy = data['policy']
    else:
        policy = data['exploration_policy']
    policy.train(False)
    if args.pause:
        import ipdb; ipdb.set_trace()

    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.mode:
        env.mode(args.mode)

    while True:
        paths = []
        for _ in range(args.nrolls):
            if args.silent:
                goal = None
            else:
                goal = env.sample_goal_for_rollout()
            path = multitask_rollout(
                env,
                policy,
                init_tau=max_tau,
                goal=goal,
                max_path_length=args.H,
                # animated=not args.hide,
                cycle_tau=args.cycle or not args.ndc,
                decrement_tau=args.dt or not args.ndc,
                env_samples_goal_on_reset=args.silent,
                # get_action_kwargs={'deterministic': True},
            )
            print("last state", path['next_observations'][-1][21:24])
            paths.append(path)
        env.log_diagnostics(paths)
        for key, value in get_generic_path_information(paths).items():
            logger.record_tabular(key, value)
        logger.dump_tabular()
