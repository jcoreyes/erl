from railrl.policies.base import ExplorationPolicy
from railrl.torch.core import PyTorchModule
import numpy as np
import railrl.torch.pytorch_util as ptu
from torch.optim import Adam


class ImplicitMPCController(PyTorchModule, ExplorationPolicy):
    def __init__(
            self,
            env,
            tdm,
            policy,
            num_simulated_paths=512,
            feasibility_weight=1,
    ):
        """
        :param env: Must implement a cost_fn of the form:

        ```
        def cost_fn(self, states, actions, next_states):
            :param states:  (BATCH_SIZE x state_dim) numpy array
            :param actions:  (BATCH_SIZE x action_dim) numpy array
            :param next_states:  (BATCH_SIZE x state_dim) numpy array
            :return: (BATCH_SIZE, ) numpy array
        ```
        :param num_simulated_paths: How many rollouts to do internally.
        """
        self.quick_init(locals())
        super().__init__()
        self.env = env
        self.tdm = tdm
        self.policy = policy
        self.num_simulated_paths = num_simulated_paths
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_dim = self.env.action_space.low.shape[0]
        self.feasibility_weight = feasibility_weight

    def forward(self, *input):
        raise NotImplementedError()

    def expand_np_to_var(self, array):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_simulated_paths,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=False)

    def expand_np(self, array):
        return np.repeat(
            np.expand_dims(array, 0),
            self.num_simulated_paths,
            axis=0
        )

    def sample_goals(self):
        return self.env.sample_states(self.num_simulated_paths)

    def sample_actions(self):
        return np.random.uniform(
            self.action_low,
            self.action_high,
            (self.num_simulated_paths, self.action_dim)
        )

    def get_feasible_actions_and_goal_states(self, obs):
        obs = self.expand_np_to_var(obs)
        goal_states = ptu.np_to_var(self.sample_goals(), requires_grad=True)
        actions = ptu.np_to_var(self.sample_actions(), requires_grad=True)
        taus = self.expand_np_to_var(np.array([0]))
        optimizer = Adam([goal_states, actions])
        for _ in range(10):
            # actions = self.policy(obs, goal_states, taus)
            cost = - self.tdm(obs, actions, goal_states, taus).mean()
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
        return ptu.get_numpy(actions), ptu.get_numpy(goal_states)

    def get_action(self, obs):
        actions, goal_states = self.get_feasible_actions_and_goal_states(obs)
        # obs = self.expand_np(obs)
        # taus = self.expand_np_to_var(np.array([0]))
        # goal_states = self.sample_goals()
        # if self.policy is None:
        #     actions = self.sample_actions()
        # else:
        #     actions = ptu.get_numpy(self.policy(
        #         ptu.np_to_var(obs),
        #         goal_states,
        #         taus,
        #     ))
        env_cost = self.env.cost_fn(obs, actions, goal_states)
        env_cost = np.expand_dims(env_cost, 1)
        # feasibility_cost = (
        #     self.tdm.eval_np(obs, actions, goal_states, taus)
        # )
        feasibility_cost = 0
        costs = env_cost + feasibility_cost * self.feasibility_weight
        min_i = np.argmin(costs)
        print("acted")
        return actions[min_i, :], {}
