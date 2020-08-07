"""
See how good a state-distance q function by comparing

    s_{QF} = \argmax_{s_g} Q(s, a s_g)

with s' from the replay buffer.
"""

import argparse

import joblib
import numpy as np
import torch
from torch.autograd import Variable
from torch.optim import Adam

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.multitask.multitask_env import MultitaskEnv
from rlkit.envs.multitask.reacher_env import (
    XyMultitaskSimpleStateReacherEnv,
    position_from_angles,
)
from rlkit.pythonplusplus import line_logger


def to_var(array):
    return Variable(
        ptu.from_numpy(array).float(),
        requires_grad=False,
    )


def check_qf(qf, replay_buffer: SplitReplayBuffer, env: MultitaskEnv):
    replay_buffer = replay_buffer.train_replay_buffer
    batch = replay_buffer.random_batch(1)
    state = to_var(batch['observations'])
    action = to_var(batch['actions'])
    target_goal_state_np = batch['next_observations']
    if type(env) == XyMultitaskSimpleStateReacherEnv:
        target_goal_state_np = position_from_angles(target_goal_state_np)

    best_goal_state = Variable(
        torch.zeros(1, env.goal_dim),
        requires_grad=True,
    )
    discount = Variable(torch.zeros(1, 1))

    lr = 1e-1
    min_num_steps_between_drops = 1000
    optim = Adam([best_goal_state], lr=lr)

    last_loss = np.inf
    last_drop = 0
    for i in range(10000):
        optim.zero_grad()
        loss = -qf(state, action, best_goal_state, discount)
        loss.backward()
        optim.step()

        best_goal_state_np = ptu.get_numpy(best_goal_state)

        difference = best_goal_state_np - target_goal_state_np
        loss = np.linalg.norm(difference)
        line_logger.print_over(
            "Distance to true next state:",
            loss,
        )
        if last_loss < loss and i - last_drop > min_num_steps_between_drops:
            lr *= 0.5
            last_drop = i
            line_logger.newline()
            print("Lowering LR to", lr)
            optim = Adam([best_goal_state], lr=lr)
        last_loss = loss
    line_logger.newline()
    print("Difference:", difference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--grid', action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    print("Done loading file")

    env = data['env']
    qf = data['qf']
    qf.train(False)
    replay_buffer = data['replay_buffer']

    check_qf(qf, replay_buffer, env)
