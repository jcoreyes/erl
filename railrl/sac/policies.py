from railrl.networks.base import Mlp
from railrl.policies.base import Policy


class TanhGaussianPolicy(Mlp, Policy):
    def get_action(self, obs):
        self.foo = None
        pass

    def forward(self, *input):
        pass

