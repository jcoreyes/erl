"""
Choose action according to

a = argmax_{a, s'} r(s, a, s') s.t. Q(s, a, s') = 0

where r is defined specifically for the reacher env.
"""

import math
import argparse

import joblib
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import SGD, Adam

import railrl.torch.pytorch_util as ptu
from railrl.envs.multitask.reacher_env import XyMultitaskSimpleStateReacherEnv
from railrl.pythonplusplus import line_logger
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


class SampleOptimalControlPolicy(object):
    """
    Do the argmax by sampling a bunch of states and acitons
    """
    R1 = 0.1  # from reacher.xml
    R2 = 0.11

    def __init__(self, qf, constraint_weight=10, sample_size=100):
        self.qf = qf
        self.constraint_weight = constraint_weight
        self._goal_pos = None
        self.sample_size = sample_size

    def set_goal(self, goal):
        self._goal = ptu.Variable(ptu.from_numpy(
            np.expand_dims(goal, 0).repeat(self.sample_size, 0)
        ).float())
        self._goal_pos = self.position(self._goal)

    def reward(self, state, action, next_state):
        ee_pos = self.position(next_state)
        return -torch.norm(ee_pos - self._goal_pos)

    def position(self, obs):
        c1 = obs[:, 0:1]  # cosine of angle 1
        c2 = obs[:, 1:2]
        s1 = obs[:, 2:3]
        s2 = obs[:, 3:4]
        return (  # forward kinematics equation for 2-link robot
            self.R1 * torch.cat((c1, s1), dim=1)
            + self.R2 * torch.cat(
                (
                    c1 * c2 - s1 * s2,
                    s1 * c2 + c1 * s2,
                ),
                dim=1,
            )
        )

    def reset(self):
        pass

    def get_action(self, obs):
        """
        Naive implementation where I just do gradient ascent on

            f(a, s') = r(s, a, s') - lambda Q(s, a, s')^2

        i.e. gradient descent on

            f(a, s') = lambda Q(s, a, s')^2 - r(s, a, s')

        :param obs: np.array, state/observation
        :return: np.array, action to take
        """
        theta = ptu.Variable(
            np.pi * (2 * torch.rand(self.sample_size, 2) - 1),
            requires_grad=True,
        )
        velocity = ptu.Variable(
            2 * torch.rand(self.sample_size, 2) - 1,
            requires_grad=True,
            )
        sampled_actions = np.random.uniform(-.2, .2, size=(self.sample_size, 2))
        action = ptu.Variable(
            ptu.from_numpy(sampled_actions).float(),
            requires_grad=True,
        )
        obs_expanded = np.expand_dims(obs, 0).repeat(self.sample_size, 0)
        obs = Variable(ptu.from_numpy(obs_expanded).float(),
                       requires_grad=False)
        next_state = torch.cat(
            (
                torch.cos(theta),
                torch.sin(theta),
                velocity,
            ),
            dim=1,
        )
        objective_loss = -self.reward(obs, action, next_state)
        augmented_obs = torch.cat((obs, next_state), dim=1)
        q_value = self.qf(augmented_obs, action)
        constraint_loss = q_value.sum()**2
        loss = (
            self.constraint_weight * constraint_loss
            + objective_loss
        )
        min_i = np.argmin(ptu.get_numpy(loss))
        return sampled_actions[min_i], {}


class GDOptimalControlPolicy(object):
    """
    Do the argmax with a gradient descent method.
    """
    R1 = 0.1  # from reacher.xml
    R2 = 0.11

    def __init__(self, qf, constraint_weight=10):
        self.qf = qf
        self.constraint_weight = constraint_weight
        self._goal_pos = None

    def set_goal(self, goal):
        self._goal = ptu.Variable(ptu.from_numpy(
            np.expand_dims(goal, 0)
        ).float())
        self._goal_pos = self.position(self._goal)

    def reward(self, state, action, next_state):
        # return -torch.norm(next_state[:, :4] - self._goal[:, :4])
        ee_pos = self.position(next_state)
        return -torch.norm(ee_pos - self._goal_pos)

    def position(self, obs):
        c1 = obs[:, 0:1]  # cosine of angle 1
        c2 = obs[:, 1:2]
        s1 = obs[:, 2:3]
        s2 = obs[:, 3:4]
        return (  # forward kinematics equation for 2-link robot
            self.R1 * torch.cat((c1, s1), dim=1)
            + self.R2 * torch.cat(
                (
                    c1 * c2 - s1 * s2,
                    s1 * c2 + c1 * s2,
                ),
                dim=1,
            )
        )

    def reset(self):
        pass

    def get_action(self, obs):
        """
        Naive implementation where I just do gradient ascent on

            f(a, s') = r(s, a, s') - lambda Q(s, a, s')^2

        i.e. gradient descent on

            f(a, s') = lambda Q(s, a, s')^2 - r(s, a, s')

        :param obs: np.array, state/observation
        :return: np.array, action to take
        """
        theta = ptu.Variable(
            np.pi * (2 * torch.rand(1, 2) - 1),
            requires_grad=True,
        )
        velocity = ptu.Variable(
            2 * torch.rand(1, 2) - 1,
            requires_grad=True,
        )
        action = ptu.Variable(
            2 * torch.rand(1, 2) - 1,
            requires_grad=True,
        )
        obs_expanded = np.expand_dims(obs, 0)
        obs = Variable(ptu.from_numpy(obs_expanded).float(),
                       requires_grad=False)
        # optimizer = SGD(
        optimizer = Adam(
            [action, theta, velocity],
            lr=1e-2,
        )
        for _ in range(1000):
            next_state = torch.cat(
                (
                    torch.cos(theta),
                    torch.sin(theta),
                    velocity,
                ),
                dim=1,
            )
            objective_loss = -self.reward(obs, action, next_state)
            augmented_obs = torch.cat((obs, next_state), dim=1)
            q_value = self.qf(augmented_obs, action)
            constraint_loss = q_value.sum()**2
            loss = (
                self.constraint_weight * constraint_loss
                + objective_loss
            )
            # loss = objective_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            action.data = torch.clamp(action.data, -1, 1)
            # loss_np = ptu.get_numpy(loss)[0]
            # if loss_np < 1:
            #     break
        #     line_logger.print_over("Loss = {}".format(ptu.get_numpy(loss)[0]))
        # line_logger.newline()
        #
        # print("")
        # print("constraint loss", ptu.get_numpy(constraint_loss)[0])
        # print("objective loss", ptu.get_numpy(objective_loss)[0])
        # print("action", ptu.get_numpy(action))
        # next_state_np = ptu.get_numpy(next_state)
        # print("next_state", next_state_np)
        # print("next_state_pos", ptu.get_numpy(self.position(next_state)))
        # print("goal_pos", ptu.get_numpy(self._goal_pos))
        return ptu.get_numpy(action), {}


def rollout(env, agent, max_path_length=np.inf, animated=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
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
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)
    print("Env type:", type(env))

    resolution = 10
    policy = SampleOptimalControlPolicy(
        qf,
        constraint_weight=1,
        sample_size=1000,
    )
    for _ in range(args.num_rollouts):
        paths = []
        for _ in range(5):
            goal = env.sample_goal_states(1)[0]
            c1 = goal[0:1]
            c2 = goal[1:2]
            s1 = goal[2:3]
            s2 = goal[3:4]
            print("Goal = ", goal)
            print("angle 1 (degrees) = ", np.arctan2(s1, c1) / math.pi * 180)
            print("angle 2 (degrees) = ", np.arctan2(s2, c2) / math.pi * 180)
            env.set_goal(goal)
            policy.set_goal(goal)
            paths.append(rollout(
                env,
                policy,
                max_path_length=args.H,
                animated=not args.hide,
            ))
        env.log_diagnostics(paths)
        logger.dump_tabular()
