from railrl.policies.base import Policy
from railrl.torch.core import PyTorchModule
import railrl.torch.pytorch_util as ptu


class MPCController(PyTorchModule, Policy):
    def __init__(
            self,
            dynamics_model,
            cost_fn,
            num_simulated_paths=10000,
            mpc_horizon=15,
    ):
        self.quick_init(locals())
        super().__init__()
        self.dynamics_model = dynamics_model
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.mpc_horizon = mpc_horizon

    def forward(self, *input):
        raise NotImplementedError()

    def get_action(self, obs):
        sampled_actions = self.env.sample_actions(self.sample_size)
        first_sampled_action = sampled_actions.copy()
        action = ptu.np_to_var(sampled_actions)
        next_obs = self.expand_np_to_var(obs)
        costs = None
        for i in range(self.planning_horizon):
            curr_obs = next_obs
            if i > 0:
                sampled_actions = self.env.sample_actions(self.sample_size)
                action = ptu.np_to_var(sampled_actions)
            next_obs = curr_obs + self.model(curr_obs, action)
            if costs is None:
                costs = self.cost_fn(
                    curr_obs,
                    action,
                    next_obs,
                )
            else:
                costs = costs + self.cost_fn(
                    curr_obs,
                    action,
                    next_obs,
                )
        min_i = np.argmin(costs)
        return first_sampled_action[min_i], {}
