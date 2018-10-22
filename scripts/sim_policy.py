from railrl.envs.remote import RemoteRolloutEnv
from railrl.samplers.util import rollout
from railrl.torch.core import PyTorchModule
import railrl.torch.pytorch_util as ptu
import argparse
import pickle
import uuid
from railrl.core import logger

filename = str(uuid.uuid4())


def simulate_policy(args):
    data = pickle.load(open(args.file, "rb"))
    if 'eval_policy' in data:
        policy = data['eval_policy']
    elif 'policy' in data:
        policy = data['policy']
    elif 'exploration_policy' in data:
        policy = data['exploration_policy']
    elif 'naf_policy' in data:
        policy = data['naf_policy']
    elif 'optimizable_qfunction' in data:
        qf = data['optimizable_qfunction']
        policy = qf.implicit_policy
    else:
        raise Exception("No policy found in loaded dict. Keys: {}".format(
            data.keys()
        ))

    env = data['env']
    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")

    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()
    if args.multitaskpause:
        env.pause_on_goal = True

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
        paths.append(rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        ))
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths, logger)
        logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    parser.add_argument('--multitaskpause', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
