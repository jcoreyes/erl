import argparse

import joblib
import numpy as np
from torch.autograd import Variable

import railrl.torch.pytorch_util as ptu
from railrl.algos.state_distance.state_distance_q_learning import (
    rollout_with_goal,
    multitask_rollout,
)
from railrl.envs.multitask.reacher_env import (
    FullStateVaryingWeightReacherEnv,
    FullStateWithXYStateReacherEnv,
)
# from railrl.samplers.util import rollout
from railrl.torch.pytorch_util import set_gpu_mode
from rllab.misc import logger


class SamplePolicy(object):
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
        sampled_actions = np.random.uniform(-.2, .2, size=(self.num_samples, 2))
        actions = Variable(ptu.from_numpy(sampled_actions).float(), requires_grad=False)
        q_values = ptu.get_numpy(self.qf(
            self.expand_np_to_var(obs),
            actions,
            self.expand_np_to_var(goal),
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
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--load', action='store_true')
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

    num_samples = 1000
    resolution = 10
    if args.load:
        policy = data['policy']
        policy.train(False)
    else:
        policy = SamplePolicy(qf, num_samples)
    while True:
        paths = []
        for _ in range(args.num_rollouts):
            goals = env.sample_goal_states(1)
            goal = goals[0]
            if isinstance(env, FullStateVaryingWeightReacherEnv):
                goal[:6] = np.array([1, 1, 1, 1, 0, 0])
            if isinstance(env, FullStateWithXYStateReacherEnv):
                goal[4:6] = 0
            if args.verbose:
                env.print_goal_state_info(goal)
            env.set_goal(goal)
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
