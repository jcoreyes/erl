"""
Greedy-action-partial-state implementation.

See https://paper.dropbox.com/doc/State-Distance-QF-Results-Summary-flRwbIxt0bbUbVXVdkKzr
for details.
"""
import sys
import math
import argparse

import joblib
import numpy as np
from torch.autograd import Variable

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import multitask_rollout
from railrl.envs.multitask.pusher import MultitaskPusherEnv
from railrl.envs.multitask.reacher_env import (
    XyMultitaskSimpleStateReacherEnv,
    FullStateVaryingWeightReacherEnv,
)
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


class SamplePolicyFixedJoints(object):
    def __init__(self, qf, num_samples):
        self.qf = qf
        self.num_samples = num_samples

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_samples,
            axis=0
        )
        return Variable(
            ptu.from_numpy(array_expanded).float(),
            requires_grad=False,
        )

    def get_action(self, obs, goal, discount):
        sampled_actions = np.random.uniform(-1, 1, size=(self.num_samples, 7))
        actions = Variable(
            ptu.from_numpy(sampled_actions).float(), requires_grad=False
        )

        goals = np.repeat(
            np.expand_dims(goal, 0),
            self.num_samples,
            axis=0
        )
        # Measured from replay buffer
        for i, (low, high) in enumerate([
            (-0.28472558315880875, 0.36561954809405517),
            (-0.4718939819254766, 0.45514972913137725),
            (-1.5033774519249183, 1.7055915060336158),
            (-2.3248252912805247, 0.0043600533992901253),
            (-1.5047841118593368, 1.5036675959237846),
            (-1.099866645233599, 0.0054087358645977099),
            (-1.5050964943586289, 1.5050477817057106),
        ]):
            goals[:, i] = np.random.uniform(low, high, size=self.num_samples)
        for i, (low, high) in enumerate([
            (-0.10531293345916685, 0.10883284387224312),
            (-0.15225655347187894, 0.13811333341407767),
            (-1.2503692354309233, 0.85581514990279628),
            (-0.76009712413916597, 0.67329304610810414),
            (-0.99178515920102206, 0.97565020741241737),
            (-0.95356149442022953, 0.97949370412197234),
            (-1.1110190330826428, 0.96856461839172359),
        ]):
            goals[:, 7+i] = np.random.uniform(low, high, size=self.num_samples)
        goals = Variable(
            ptu.from_numpy(goals).float(),
            requires_grad=False,
        )

        q_values = ptu.get_numpy(self.qf(
            self.expand_np_to_var(obs),
            actions,
            goals,
            self.expand_np_to_var(np.array([discount])),
        ))
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
                        help='Number of rollouts per eval')
    parser.add_argument('--discount', type=float, default=0.,
                        help='Discount Factor')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    if type(env) != MultitaskPusherEnv:
        print("Only works for multitask pusher env")
        print("Exiting...")
        sys.exit()
    qf = data['qf']
    if args.gpu:
        set_gpu_mode(True)
        qf.cuda()
    qf.train(False)

    num_samples = 10000
    policy = SamplePolicyFixedJoints(qf, num_samples)

    while True:
        paths = []
        for _ in range(args.num_rollouts):
            goals = env.sample_goal_states(1)
            goal = goals[0]
            # if args.verbose:
            #     env.print_goal_state_info(goal)
            # env.set_goal(goal)
            path = multitask_rollout(
                env,
                policy,
                goal,
                discount=args.discount,
                max_path_length=args.H,
                animated=not args.hide,
            )
            path['goal_states'] = goals.repeat(len(path['observations']), 0)
            paths.append(path)
        env.log_diagnostics(paths)
        logger.dump_tabular()
