import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from railrl.networks.base import Mlp
from railrl.policies.base import Policy
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from rllab.misc import logger


class AmortizedPolicy(PyTorchModule, Policy):
    def __init__(
            self,
            goal_reaching_policy,
            goal_chooser,
            discount,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.goal_reaching_policy = goal_reaching_policy
        self.goal_chooser = goal_chooser
        self._discount_expanded_torch = ptu.np_to_var(
            np.array([[discount]])
        )

    def get_action(self, obs_np):
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        goal = self.goal_chooser(obs)
        # print("Goal chosen: {}".format(ptu.get_numpy(goal)))
        action = self.goal_reaching_policy(
            obs,
            goal,
            self._discount_expanded_torch,
        )
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}


class ReacherGoalChooser(Mlp):
    def __init__(
            self,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            input_size=6,
            output_size=4,
            output_activation=None,
            **kwargs
        )

    def forward(self, input):
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        output = self.last_fc(h)
        output_theta_pre_activation = output[:, :2]
        theta = np.pi * F.tanh(output_theta_pre_activation)
        output_vel = output[:, 2:]

        return torch.cat(
            (
                torch.cos(theta),
                torch.sin(theta),
                output_vel,
            ),
            dim=1
        )


def train_amortized_goal_chooser(
        goal_chooser,
        goal_conditioned_model,
        argmax_q,
        rewards_py_fctn,
        discount,
        replay_buffer,
        learning_rate=1e-3,
        batch_size=32,
        num_updates=1000,
):
    def get_loss(training=False):
        buffer = replay_buffer.get_replay_buffer(training)
        obs = buffer.random_batch(batch_size)['observations']
        obs = ptu.np_to_var(obs, requires_grad=False)
        goal = goal_chooser(obs)
        actions = argmax_q(
            obs,
            goal,
            discount
        )
        final_state_predicted = goal_conditioned_model(
            obs,
            actions,
            goal,
            discount,
        ) + obs
        rewards = rewards_py_fctn(final_state_predicted)
        return -rewards.mean()

    discount = ptu.np_to_var(discount * np.ones((batch_size, 1)))
    optimizer = optim.Adam(goal_chooser.parameters(), learning_rate)
    for i in range(num_updates):
        optimizer.zero_grad()
        loss = get_loss()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            logger.log("Number updates: {}".format(i))
            logger.log("Train loss: {}".format(
                float(ptu.get_numpy(loss)))
            )
            logger.log("Validation loss: {}".format(
                float(ptu.get_numpy(get_loss(training=False))))
            )
