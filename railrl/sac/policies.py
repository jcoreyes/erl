import torch
from railrl.torch.networks import Mlp
from railrl.policies.base import Policy
import railrl.torch.pytorch_util as ptu


class TanhGaussianPolicy(Mlp, Policy):
    def get_action(self, obs):
        self.foo = None
        obs = ptu.np_to_var(
            np.expand_dims(obs_np, 0)
        )
        action = self.__call__(
            obs,
            self._goal_expanded_torch,
            self._discount_expanded_torch,
        )
        mean = action.squeeze(0)
        action_np = ptu.get_numpy(action), {}
        action_np =

        return action_np, {}

    def forward(self, input, return_log_prob=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        if return_log_prob:


        else:
            return self.output_activation(self.last_fc(h))

