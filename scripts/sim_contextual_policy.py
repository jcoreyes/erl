import argparse
import pickle

from railrl.core import logger
from railrl.envs.remote import RemoteRolloutEnv
from railrl.envs.vae_wrappers import VAEWrappedEnv
from railrl.samplers.rollout_functions import (
    multitask_rollout,
    contextual_rollout,
)
from railrl.torch.core import PyTorchModule
import railrl.torch.pytorch_util as ptu


def simulate_policy(args):
    if args.pause:
        import ipdb; ipdb.set_trace()
    data = pickle.load(open(args.file, "rb")) # joblib.load(args.file)
    if 'policy' in data:
        policy = data['policy']
    elif 'evaluation/policy' in data:
        policy = data['evaluation/policy']

    if 'env' in data:
        env = data['env']
    elif 'evaluation/env' in data:
        env = data['evaluation/env']

    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.to(ptu.device)
    else:
        ptu.set_gpu_mode(False)
        policy.to(ptu.device)
    if isinstance(env, VAEWrappedEnv):
        env.mode(args.mode)
    if args.enable_render or hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.multitaskpause:
        env.pause_on_goal = True
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    paths = []
    while True:
        paths.append(contextual_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
            observation_key=data.get('evaluation/observation_key', 'observation'),
            # desired_goal_key=data.get('evaluation/desired_goal_key', 'desired_goal'),
            context_keys_for_policy=['state_desired_goal'],
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
        if hasattr(env, "get_diagnostics"):
            for k, v in env.get_diagnostics(paths).items():
                logger.record_tabular(k, v)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--mode', default='video_env', type=str,
                        help='env mode')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
