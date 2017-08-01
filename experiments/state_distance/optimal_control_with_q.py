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
from railrl.envs.multitask.reacher_env import XyMultitaskSimpleStateReacherEnv, \
    GoalStateSimpleStateReacherEnv
from railrl.pythonplusplus import line_logger
from railrl.samplers.util import rollout
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


class SampleOptimalControlPolicy(object):
    """
    Do the argmax by sampling a bunch of states and acitons
    """
    R1 = 0.1  # from reacher.xml
    R2 = 0.11

    def __init__(
            self,
            qf,
            constraint_weight=10,
            sample_size=100,
            goal_is_full_state=True,
            verbose=False,
    ):
        self.qf = qf
        self.constraint_weight = constraint_weight
        self._goal_pos = None
        self.sample_size = sample_size
        self.goal_is_full_state = goal_is_full_state
        self.verbose = verbose

    def set_goal(self, goal):
        if self.goal_is_full_state:
            self._goal = ptu.Variable(ptu.from_numpy(
                np.expand_dims(goal, 0).repeat(self.sample_size, 0)
            ).float())
            self._goal_pos = self.position(self._goal)
        else:
            self._goal_pos = ptu.Variable(ptu.from_numpy(
                np.expand_dims(goal, 0).repeat(self.sample_size, 0)
            ).float())

    def reward(self, state, action, next_state):
        ee_pos = self.position(next_state)
        return -torch.norm(ee_pos - self._goal_pos, dim=1)

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
        Naive implementation where I just sample a bunch of a and s' and take
        the one that maximizes

            f(a, s') = r(s, a, s') - C * Q_d(s, a, s')**2

        :param obs: np.array, state/observation
        :return: np.array, action to take
        """
        theta = ptu.Variable(
            np.pi * (2 * torch.rand(self.sample_size, 2) - 1),
            requires_grad=True,
        )
        velocity = ptu.Variable(
            5 * torch.rand(self.sample_size, 2) - 1,
            requires_grad=True,
        )
        sampled_actions = np.random.uniform(-1, 1, size=(self.sample_size, 2))
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
        reward = self.reward(obs, action, next_state)
        if self.goal_is_full_state:
            augmented_obs = torch.cat((obs, next_state), dim=1)
        else:
            augmented_obs = torch.cat((
                obs,
                self.position(next_state),
            ), dim=1)
        constraint_penalty = self.qf(augmented_obs, action)**2
        score = (
            reward
            - self.constraint_weight * constraint_penalty
        )
        max_i = np.argmax(ptu.get_numpy(score))
        if self.verbose:
            print("")
            print("constraint penalty", ptu.get_numpy(constraint_penalty)[max_i])
            print("reward", ptu.get_numpy(reward)[max_i])
            print("action", ptu.get_numpy(action)[max_i])
            print("next_state", ptu.get_numpy(next_state[max_i]))
            print("next_state_pos", ptu.get_numpy(self.position(next_state))[max_i])
            print("goal_pos", ptu.get_numpy(self._goal_pos)[max_i])
        return sampled_actions[max_i], {}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)
    print("Env type:", type(env))
    goal_is_full_state = isinstance(env, GoalStateSimpleStateReacherEnv)

    policy = SampleOptimalControlPolicy(
        qf,
        constraint_weight=10,
        sample_size=1000,
        goal_is_full_state=goal_is_full_state,
        verbose=args.verbose,
    )
    for _ in range(args.num_rollouts):
        paths = []
        for _ in range(5):
            goals = env.sample_goal_states(1)
            goal = goals[0]
            c1 = goal[0:1]
            c2 = goal[1:2]
            s1 = goal[2:3]
            s2 = goal[3:4]
            print("Goal = ", goal)
            print("angle 1 (degrees) = ", np.arctan2(s1, c1) / math.pi * 180)
            print("angle 2 (degrees) = ", np.arctan2(s2, c2) / math.pi * 180)
            print("angle 1 (radians) = ", np.arctan2(s1, c1))
            print("angle 2 (radians) = ", np.arctan2(s2, c2))
            env.set_goal(goal)
            policy.set_goal(goal)
            path = rollout(
                env,
                policy,
                max_path_length=args.H,
                animated=not args.hide,
            )
            path['observations'] = np.hstack((
                path['observations'],
                goals.repeat(len(path['observations']), 0),
            ))
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()
