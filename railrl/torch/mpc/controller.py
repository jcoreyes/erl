import numpy as np
from torch import optim

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
        Optimization is done by a shooting method.

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


# GOAL_SLICE = slice(0, 2)
GOAL_SLICE = slice(0, 7)


class GradientBasedMPCController(PyTorchModule, ExplorationPolicy):
    """
    Optimization is done with gradient descent
    """

    def __init__(
            self,
            env,
            dynamics_model,
            mpc_horizon=15,
            learning_rate=1e-1,
            num_grad_steps=10,
    ):
        """
        Optimization is done by a shooting method.

        :param env:
        :param dynamics_model: Dynamics model. See dagger/model.py
        :param mpc_horizon: How long to plan for.
        """
        assert mpc_horizon >= 1
        self.quick_init(locals())
        super().__init__()
        self.env = env
        self.dynamics_model = dynamics_model
        self.mpc_horizon = mpc_horizon
        self.action_low_repeated = np.repeat(
            self.env.action_space.low,
            self.mpc_horizon,
        )
        self.action_high_repeated = np.repeat(
            self.env.action_space.high,
            self.mpc_horizon
        )
        self.action_dim = self.env.action_space.low.shape[0]
        self.last_actions_np = None
        self.learning_rate = learning_rate
        self.num_grad_steps = num_grad_steps

    def forward(self, *input):
        raise NotImplementedError()

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_simulated_paths,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def reset(self):
        self.last_actions_np = None

    def cost_function(self, states, all_actions):
        """
        Everything is batch-wise.
        """
        loss = 0
        for i in range(self.mpc_horizon):
            actions = (
                all_actions[:, i * self.action_dim:(i + 1) * self.action_dim]
            )
            next_states = states + self.dynamics_model(states, actions)
            next_features_predicted = next_states[:, GOAL_SLICE]
            desired_features = ptu.np_to_var(
                self.env.multitask_goal[GOAL_SLICE][None]
                * np.ones(next_features_predicted.shape)
            )
            diff = next_features_predicted - desired_features
            loss += (diff ** 2).sum(dim=1, keepdim=True)
        return loss

    def get_action(self, obs):
        # if self.last_actions_np is None:
        if True:
            init_actions = np.hstack([
                self.env.action_space.sample()
                for _ in range(self.mpc_horizon)
            ])
        else:
            init_actions = self.last_actions_np
        all_actions = ptu.np_to_var(init_actions[None], requires_grad=True)
        obs = ptu.np_to_var(obs[None])
        optimizer = optim.Adam([all_actions], lr=self.learning_rate)
        print("--")
        for i in range(self.num_grad_steps):
            print(ptu.get_numpy(all_actions)[0, :self.action_dim])
            loss = self.cost_function(obs, all_actions)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        self.last_actions_np = np.clip(
            ptu.get_numpy(all_actions)[0],
            self.action_low_repeated,
            self.action_high_repeated,
        )
        action = self.last_actions_np[:self.action_dim]
        return action, {}
