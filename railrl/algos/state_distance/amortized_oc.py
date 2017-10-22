import numpy as np
from torch import optim

from railrl.policies.base import Policy
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule

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
        print("Goal chosen: {}".format(ptu.get_numpy(goal)))
        action = self.goal_reaching_policy(
            obs,
            goal,
            self._discount_expanded_torch,
        )
        action = action.squeeze(0)
        return ptu.get_numpy(action), {}


def train_amortized_goal_chooser(
        goal_chooser,
        goal_conditioned_model,
        argmax_q,
        env,
        rewards_py_fctn,
        discount,
        replay_buffer,
        learning_rate=1e-3
):
    batch_size = 32
    discount = ptu.np_to_var(discount * np.ones((batch_size, 1)))
    optimizer = optim.Adam(goal_chooser.parameters(), learning_rate)
    for _ in range(1000):
        obs = replay_buffer.random_batch(batch_size)['observations']
        # obs = env.sample_states(batch_size)
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

        optimizer.zero_grad()
        loss = -rewards.mean()
        loss.backward()
        optimizer.step()
