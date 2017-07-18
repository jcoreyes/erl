import argparse

import joblib
import numpy as np
from torch.autograd import Variable

import railrl.torch.pytorch_util as ptu
from railrl.torch.pytorch_util import set_gpu_mode


class SamplePolicy(object):
    def __init__(self, qf, num_samples):
        self.qf = qf
        self.num_samples = num_samples

    def get_action(self, obs):
        sampled_actions = np.random.uniform(-.2, .2, size=(self.num_samples, 2))
        obs_expanded = np.repeat(
            np.expand_dims(obs, 0),
            self.num_samples,
            axis=0
        )
        actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
        obs = Variable(ptu.from_numpy(obs_expanded).float(), requires_grad=False)
        q_values = ptu.get_numpy(self.qf(obs, actions))
        max_i = np.argmax(q_values)
        return sampled_actions[max_i]


class GridPolicy(object):
    def __init__(self, qf, resolution):
        self.qf = qf
        self.resolution = resolution

    def get_action(self, obs):
        x = np.linspace(-1, 1, self.resolution)
        y = np.linspace(-1, 1, self.resolution)
        sampled_actions = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        num_samples = resolution**2
        obs_expanded = np.repeat(np.expand_dims(obs, 0), num_samples, axis=0)
        actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
        obs = Variable(ptu.from_numpy(obs_expanded).float(), requires_grad=False)
        q_values = ptu.get_numpy(self.qf(obs, actions))
        max_i = np.argmax(q_values)
        # vals = q_values.reshape(resolution, resolution)
        # heatmap = vals, x, y, _
        # fig, ax = plt.subplots(1, 1)
        # plot_heatmap(fig, ax, heatmap)
        # plt.show()
        return sampled_actions[max_i]


def rollout(env, agent, goal, max_path_length=np.inf, animated=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    o = np.hstack((o, goal))
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        o = np.hstack((o, goal))
        if animated:
            env.render()

    return dict(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards),
        terminals=np.array(terminals),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--use_policy', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)

    num_samples = 1000
    resolution = 10
    if args.use_policy:
        policy = data['policy']
    else:
        if args.grid:
            policy = GridPolicy(qf, resolution)
        else:
            policy = SamplePolicy(qf, num_samples)

    for _ in range(args.num_rollouts):
        goal = env.sample_goal_states(1)[0]
        env.set_goal(goal)
        path = rollout(
            env,
            policy,
            goal,
            max_path_length=args.max_path_length,
            animated=True,
        )
        env.log_diagnostics([path])
