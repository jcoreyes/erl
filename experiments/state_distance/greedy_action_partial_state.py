"""
Greedy-action-partial-state implementation.

See https://paper.dropbox.com/doc/State-Distance-QF-Results-Summary-flRwbIxt0bbUbVXVdkKzr
for details.
"""
import math
import argparse

import joblib
import numpy as np
from torch.autograd import Variable

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import \
    rollout_with_goal
from railrl.envs.multitask.reacher_env import FullStateVaryingWeightReacherEnv
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


class SamplePolicyFixedJoints(object):
    def __init__(self, qf, num_samples):
        self.qf = qf
        self.num_samples = num_samples

    def get_action(self, obs):
        sampled_actions = np.random.uniform(-.2, .2, size=(self.num_samples, 2))
        sampled_velocities = np.random.uniform(-1, 1, size=(self.num_samples, 2))
        obs_expanded = np.repeat(
            np.expand_dims(obs, 0),
            self.num_samples,
            axis=0
        )
        obs_expanded[:, -2:] = sampled_velocities
        actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
        obs = Variable(ptu.from_numpy(obs_expanded).float(), requires_grad=False)
        q_values = ptu.get_numpy(self.qf(obs, actions))
        max_i = np.argmax(q_values)
        return sampled_actions[max_i], {}

    def reset(self):
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', type=int, default=100,
                        help='Total number of rollout')
    parser.add_argument('--nogpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']
    if not args.nogpu:
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)

    num_samples = 1000
    policy = SamplePolicyFixedJoints(qf, num_samples)

    for _ in range(args.num_rollouts):
        paths = []
        for _ in range(5):
            goal = env.sample_goal_states(1)[0]
            if isinstance(env, FullStateVaryingWeightReacherEnv):
                goal[:6] = np.array([1, 1, 1, 1, 0, 0])
            env.print_goal_state_info(goal)
            env.set_goal(goal)
            paths.append(rollout_with_goal(
                env,
                policy,
                goal,
                max_path_length=args.H,
                animated=not args.hide,
            ))
        env.log_diagnostics(paths)
        logger.dump_tabular()
