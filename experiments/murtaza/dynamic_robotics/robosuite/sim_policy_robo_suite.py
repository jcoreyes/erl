from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.misc import eval_util
from rlkit.samplers.rollout_functions import deprecated_rollout
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
import argparse
import pickle
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = pickle.load(open(args.file, "rb"))
    policy_key = args.policy_type+'/policy'
    if policy_key in data:
        policy = data[policy_key]
    else:
        raise Exception("No policy found in loaded dict. Keys: {}".format(
            data.keys()
        ))

    env_key = args.env_type + '/env'
    if env_key in data:
        env = data[env_key]
    else:
        raise Exception("No environment found in loaded dict. Keys: {}".format(
            data.keys()
        ))

    #robosuite env specific things
    env._wrapped_env.has_renderer = True
    env.reset()
    env.viewer.set_camera(camera_id=0)

    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")

    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.gpu:
        ptu.set_gpu_mode(True)
    if hasattr(policy, "to"):
        policy.to(ptu.device)
    if hasattr(env, "vae"):
        env.vae.to(ptu.device)

    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    paths = []
    while True:
        paths.append(deprecated_rollout(
            env,
            policy,
            max_path_length=args.H,
            render=not args.hide,
        ))
        if args.log_diagnostics:
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics(paths, logger)
            for k, v in eval_util.get_generic_path_information(paths).items():
                logger.record_tabular(k, v)
            logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--policy_type', type=str, default='evaluation')
    parser.add_argument('--env_type', type=str, default='evaluation')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--log_diagnostics', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
