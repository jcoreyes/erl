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
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    num_samples = 1000
    resolution = 10
    if 'policy' in data:
        policy = data['policy']
    else:
        policy = data['exploration_policy']
    policy.train(False)
    if args.gpu:
        ptu.set_gpu_mode(True)
        policy.cuda()
    '''
    EVAL STEPS
    - load in goal data set
    - load a particular iteration's policy as well as the env
    - iterate through each goal image
    - encode goal image to latent and set it as goal
    - execute a rollout for that goal
    - display goal image and ask for whether it succeeded or not
    - print/save the success percentage 
    '''
    while True:
        paths = []
        for _ in range(args.nrolls):
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
                # get_action_kwargs={'deterministic': True},
            )
            print("last state", path['next_observations'][-1][21:24])
            paths.append(path)
        env.log_diagnostics(paths)
        import ipdb; ipdb.set_trace()
        for key, value in get_generic_path_information(paths).items():
            logger.record_tabular(key, value)
        logger.dump_tabular()
