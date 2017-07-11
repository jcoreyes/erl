import numpy as np
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
from rllab.misc import logger
from torch.autograd import Variable
import railrl.torch.pytorch_util as ptu

filename = str(uuid.uuid4())


def sample_best_action(qf, obs, num_samples):
    x = np.linspace(-.1, .1, 10)
    y = np.linspace(-.1, .1, 10)
    sampled_actions = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    # sampled_actions = np.random.uniform(-.1, .1, size=(num_samples, 2))
    obs_expanded = np.repeat(np.expand_dims(obs, 0), num_samples, axis=0)
    actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
    obs = Variable(ptu.from_numpy(obs_expanded).float(), requires_grad=False)
    q_values = ptu.get_numpy(qf(obs, actions))
    max_i = np.argmax(q_values)
    return sampled_actions[max_i]

if __name__ == "__main__":

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
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
        qf.cuda()

    goal = np.array([.1, .1])
    obs = env.reset()
    print(obs)
    for _ in range(1000):
        new_obs = np.hstack((obs, goal))
        action = sample_best_action(qf, new_obs, 100)
        next_obs, r, d, env_info = env.step(action)
        env.render()
        obs = next_obs
