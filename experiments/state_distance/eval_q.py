import argparse

import joblib
import numpy as np
from torch.autograd import Variable

import railrl.torch.pytorch_util as ptu
from railrl.torch.pytorch_util import set_gpu_mode


def sample_best_action(qf, obs, num_samples):
    # x = np.linspace(-1, 1, resolution)
    # y = np.linspace(-1, 1, resolution)
    # sampled_actions = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    # num_samples = resolution**2
    sampled_actions = np.random.uniform(-.1, .1, size=(num_samples, 2))
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
    parser.add_argument('--truth', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.cuda()
        qf.train(False)

    goal = np.array([.1, .1])
    num_samples = 10

    obs = env.reset()
    for _ in range(args.max_path_length):
        new_obs = np.hstack((obs, goal))
        action = sample_best_action(qf, new_obs, num_samples)
        obs, r, d, env_info = env.step(action)
        env.render()
