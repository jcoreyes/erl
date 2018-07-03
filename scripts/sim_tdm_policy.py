import argparse
import json

import joblib
from pathlib import Path

import railrl.torch.pytorch_util as ptu
from railrl.envs.vae_wrappers import VAEWrappedEnv
from railrl.envs.wrappers import ImageMujocoEnv
from railrl.misc.eval_util import get_generic_path_information
from railrl.pythonplusplus import find_key_recursive
from railrl.samplers.rollout_functions import tdm_rollout
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
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--mode', type=str, help='env mode',
                        default='video_env')
    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--ndt', help='no decrement tau', action='store_true')
    parser.add_argument('--ncycle', help='no cycle tau', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
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
    is_mj_env = (
            isinstance(env, VAEWrappedEnv) and
            isinstance(env.wrapped_env, ImageMujocoEnv)
    )
    if isinstance(env, VAEWrappedEnv):
        env.mode(args.mode)
    if args.enable_render or is_mj_env:
        # some environments need to be reconfigured for visualization
        env.enable_render()

    paths = []
    env_samples_goal_on_reset = args.silent or is_mj_env
    while True:
        for _ in range(args.nrolls):
            path = tdm_rollout(
                env,
                policy,
                init_tau=max_tau,
                max_path_length=args.H,
                animated=not args.hide and not is_mj_env,
                cycle_tau=not args.ncycle,
                decrement_tau=not args.ndt,
                # observation_key='state_observation',
                # desired_goal_key='state_desired_goal',
                observation_key='latent_observation',
                desired_goal_key='latent_desired_goal',
            )
            print("last state", path['next_observations'][-1])
            paths.append(path)
        env.log_diagnostics(paths)
        for key, value in get_generic_path_information(paths).items():
            logger.record_tabular(key, value)
        logger.dump_tabular()
