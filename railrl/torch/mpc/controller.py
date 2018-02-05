import numpy as np
from railrl.policies.base import ExplorationPolicy
from railrl.torch.core import PyTorchModule
import railrl.torch.pytorch_util as ptu


class MPCController(PyTorchModule, ExplorationPolicy):
    def __init__(
            self,
            env,
            dynamics_model,
            cost_fn,
            num_simulated_paths=10000,
            mpc_horizon=15,
    ):
        """
        :param env:
        :param dynamics_model: Dynamics model. See dagger/model.py
        :param cost_fn:  Function of the form:

        ```
        def cost_fn(self, states, actions, next_states):
            :param states:  (BATCH_SIZE x state_dim) numpy array
            :param actions:  (BATCH_SIZE x action_dim) numpy array
            :param next_states:  (BATCH_SIZE x state_dim) numpy array
            :return: (BATCH_SIZE, ) numpy array
        ```
        :param num_simulated_paths: How many rollouts to do internally.
        :param mpc_horizon: How long to plan for.
        """
        assert mpc_horizon >= 1
        self.quick_init(locals())
        super().__init__()
        self.env = env
        self.dynamics_model = dynamics_model
        self.cost_fn = cost_fn
        self.num_simulated_paths = num_simulated_paths
        self.mpc_horizon = mpc_horizon
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_dim = self.env.action_space.low.shape[0]

    def forward(self, *input):
        raise NotImplementedError()

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_simulated_paths,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def sample_actions(self):
        return np.random.uniform(
            self.action_low,
            self.action_high,
            (self.num_simulated_paths, self.action_dim)
        )

    def get_action(self, obs):
        sampled_actions = self.sample_actions()
        first_sampled_actions = sampled_actions.copy()
        actions = ptu.np_to_var(sampled_actions)
        next_obs = self.expand_np_to_var(obs)
        costs = 0
        for i in range(self.mpc_horizon):
            curr_obs = next_obs
            if i > 0:
                sampled_actions = self.sample_actions()
                actions = ptu.np_to_var(sampled_actions)
            next_obs = curr_obs + self.dynamics_model(curr_obs, actions)
            costs = costs + self.cost_fn(
                ptu.get_numpy(curr_obs),
                ptu.get_numpy(actions),
                ptu.get_numpy(next_obs),
            )
        min_i = np.argmin(costs)
        return first_sampled_actions[min_i], {}
