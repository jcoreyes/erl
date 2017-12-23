import numpy as np
from railrl.policies.base import ExplorationPolicy
from railrl.torch.core import PyTorchModule
import railrl.torch.pytorch_util as ptu
import time


class MPCController(PyTorchModule, ExplorationPolicy):
    def __init__(
            self,
            env,
            dynamics_model,
            cost_fn,
            num_simulated_paths=10000,
            mpc_horizon=15,
    ):
        self.quick_init(locals())
        super().__init__()
        self.env = env
        self.dynamics_model = dynamics_model
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.mpc_horizon = mpc_horizon

    def forward(self, *input):
        raise NotImplementedError()

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_simulated_paths,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def get_action(self, obs):
        sampled_actions = self.env.sample_actions(self.num_simulated_paths)
        first_sampled_actions = sampled_actions.copy()
        actions = ptu.np_to_var(sampled_actions)
        next_obs = self.expand_np_to_var(obs)
        costs = np.zeros(self.num_simulated_paths)
        for i in range(self.mpc_horizon):
            curr_obs = next_obs
            if i > 0:
                sampled_actions = self.env.sample_actions(
                    self.num_simulated_paths
                )
                actions = ptu.np_to_var(sampled_actions)
            next_obs = curr_obs + self.dynamics_model(curr_obs, actions)
            costs = costs + self.cost_fn(
                ptu.get_numpy(curr_obs),
                ptu.get_numpy(actions),
                ptu.get_numpy(next_obs),
            )
        min_i = np.argmin(costs)
        return first_sampled_actions[min_i], {}
