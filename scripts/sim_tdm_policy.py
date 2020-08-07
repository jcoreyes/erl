import argparse
import json

import joblib
from pathlib import Path

import rlkit.torch.pytorch_util as ptu
from rlkit.envs.vae_wrappers import VAEWrappedEnv
from rlkit.pythonplusplus import find_key_recursive
from rlkit.samplers.rollout_functions import tdm_rollout
from rlkit.core import logger


def get_max_tau(args):
    if args.mtau is None:
        variant_path = Path(args.file).parents[0] / 'variant.json'
        variant = json.load(variant_path.open())
        max_tau = find_key_recursive(variant, 'max_tau')
        if max_tau is None:
            print("Defaulting max tau to 0.")
            max_tau = 0
        else:
            print("Max tau read from variant: {}".format(max_tau))
    else:
        max_tau = args.mtau
    return max_tau


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
    parser.add_argument('--mode', type=str, help='env mode',
                        default='video_env')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--ndt', help='no decrement tau', action='store_true')
    parser.add_argument('--ncycle', help='no cycle tau', action='store_true')
    args = parser.parse_args()

    max_tau = get_max_tau(args)
    data = joblib.load(args.file)

    env = data['env']
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
    if isinstance(env, VAEWrappedEnv):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()

    paths = []
    while True:
        for _ in range(args.nrolls):
            path = tdm_rollout(
                env,
                policy,
                init_tau=max_tau,
                max_path_length=args.H,
                animated=not args.hide,
                cycle_tau=not args.ncycle,
                decrement_tau=not args.ndt,
                observation_key='observation',
                desired_goal_key='desired_goal',
            )
            print("last state", path['next_observations'][-1])
            paths.append(path)
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()
