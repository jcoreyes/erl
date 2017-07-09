import numpy as np
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
from rllab.misc import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    policy = data['policy']
    qf = data['qf']
    env = data['env']
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()

    goal = np.array([-.1, .1])
    obs = env.reset()
    print(obs)
    for _ in range(1000):
        new_obs = np.hstack((obs, goal))
        # TODO(vitchyr): optimize for the best action
        action, agent_info = policy.get_action(new_obs)
        next_obs, r, d, env_info = env.step(action)
        env.render()
        obs = next_obs

if __name__ == "__main__":
    main()
