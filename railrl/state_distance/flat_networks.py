import torch
from railrl.torch.networks import Mlp


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
