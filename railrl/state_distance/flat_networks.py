import torch

from railrl.state_distance.util import split_tau
from railrl.torch.networks import Mlp
import railrl.torch.pytorch_util as ptu


class StructuredQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            output_size,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + 1,
            output_size=output_size,
            **kwargs
        )

    def forward(self, *inputs):
        h = torch.cat(inputs, dim=1)
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))


class OneHotTauQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a one-hot vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            action_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + max_tau + 1,
            output_size=output_size,
            **kwargs
        )
        self.max_tau = max_tau

    def forward(self, flat_obs, action):
        obs, taus = split_tau(flat_obs)
        h = torch.cat((obs, action), dim=1)

        batch_size = h.size()[0]
        y_onehot = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_onehot.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_onehot.scatter_(1, t, 1)

        h = torch.cat((
            obs,
            ptu.Variable(y_onehot),
            action
        ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))
