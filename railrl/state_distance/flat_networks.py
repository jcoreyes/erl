import torch
import railrl.torch.pytorch_util as ptu
from railrl.state_distance.util import split_tau
from railrl.torch.networks import Mlp, SeparateFirstLayerMlp
import numpy as np

class StructuredQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, discount) = - |f(s, a, s_g, discount)|

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
        # Keeping it as a separate argument to have same interface
        # assert observation_dim == goal_dim
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
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            action_dim=0,
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

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs
        batch_size = h.size()[0]
        y_binary = ptu.FloatTensor(batch_size, self.max_tau + 1)
        y_binary.zero_()
        t = taus.data.long()
        t = torch.clamp(t, min=0)
        y_binary.scatter_(1, t, 1)
        if action is not None:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
                action
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))

class BinaryStringTauQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a binary string vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        self.max_tau = np.unpackbits(np.array(max_tau, dtype=np.uint8))
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + len(self.max_tau),
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs
        batch_size = h.size()[0]
        y_binary = make_binary_tensor(taus, len(self.max_tau), batch_size)

        if action is not None:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),
                action
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(y_binary),

            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))

def make_binary_tensor(tensor, max_len, batch_size):
    binary = ptu.FloatTensor(batch_size, max_len)
    for i in range(batch_size):
        bin = np.unpackbits(np.array(tensor[i], dtype=np.uint8))
        bin = np.hstack((np.zeros(max_len - len(bin)), bin))
        bin = torch.from_numpy(bin)
        binary[i,:] = bin
    return binary

class TauVectorQF(Mlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a binary string vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            tau_vector_len=0,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            input_size=observation_dim + action_dim + goal_dim + self.tau_vector_len,
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs
        batch_size = h.size()[0]
        tau_vector = torch.zeros((batch_size, self.tau_vector_len)) + taus.data
        if action is not None:
            h = torch.cat((
                obs,
                ptu.Variable(tau_vector),
                action
            ), dim=1)
        else:
            h = torch.cat((
                obs,
                ptu.Variable(tau_vector),

            ), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        return - torch.abs(self.last_fc(h))

class TauVectorSeparateFirstLayerQF(SeparateFirstLayerMlp):
    """
    Parameterize QF as

    Q(s, a, s_g, tau) = - |f(s, a, s_g, tau)|

    element-wise, and represent tau as a binary string vector.

    WARNING: this is only valid for when the reward is the negative abs value
    along each dimension.
    """
    def __init__(
            self,
            observation_dim,
            goal_dim,
            output_size,
            max_tau,
            hidden_sizes,
            tau_vector_len=0,
            action_dim=0,
            **kwargs
    ):
        self.save_init_params(locals())
        if tau_vector_len == 0:
            self.tau_vector_len = max_tau
        super().__init__(
            hidden_sizes=hidden_sizes,
            first_input_size=observation_dim + action_dim + goal_dim,
            second_input_size=tau_vector_len,
            output_size=output_size,
            **kwargs
        )

    def forward(self, flat_obs, action=None):
        obs, taus = split_tau(flat_obs)
        if action is not None:
            h = torch.cat((obs, action), dim=1)
        else:
            h = obs

        batch_size = h.size()[0]
        tau_vector = torch.zeros((batch_size, self.tau_vector_len)) + taus.data
        return - torch.abs(super().forward(h, tau_vector))