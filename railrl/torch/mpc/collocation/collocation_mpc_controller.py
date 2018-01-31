import numpy as np
import torch
from scipy import optimize
from torch import optim, nn as nn

from railrl.core import logger
from railrl.policies.base import ExplorationPolicy
from railrl.state_distance.policies import UniversalPolicy
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule


class CollocationMpcController(PyTorchModule, ExplorationPolicy):
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

    def expand_np_to_var(self, array, requires_grad=False):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_simulated_paths,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=requires_grad)

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

    def get_feasible_actions_and_goal_states(self, single_obs):
        obs = self.expand_np_to_var(single_obs)
        actions = ptu.np_to_var(self.sample_actions(), requires_grad=True)
        taus = self.expand_np_to_var(np.array([0]))
        goal_states = self.expand_np_to_var(single_obs.copy(),
                                            requires_grad=True)
        optimizer = optim.RMSprop([goal_states], lr=1e-1)
        print("--")
        for _ in range(10):
            distance = -(self.tdm(obs, goal_states, taus, actions)).mean()
            print(ptu.get_numpy(distance.mean())[0])
            optimizer.zero_grad()
            distance.backward()
            optimizer.step()
        return ptu.get_numpy(actions), ptu.get_numpy(goal_states)

    def get_feasible_goal_states_and_tdm_actions(self, single_obs):
        obs = self.expand_np_to_var(single_obs)
        taus = self.expand_np_to_var(np.array([10]))
        goal_states = self.expand_np_to_var(single_obs.copy(),
                                            requires_grad=True)
        goal_states.data = (
            goal_states.data + torch.randn(goal_states.shape) * 0.05
        )

        optimizer = optim.RMSprop([goal_states], lr=1e-2)
        print("--")
        for _ in range(0):
            new_obs = torch.cat(
                (
                    obs,
                    goal_states,
                    taus,
                ),
                dim=1,
            )
            # actions = self.policy(new_obs, deterministic=True)[0]
            actions = self.policy(new_obs)
            distance = -(self.tdm(obs, goal_states, taus, actions)).mean()
            print(ptu.get_numpy(distance.mean())[0])
            optimizer.zero_grad()
            distance.backward()
            optimizer.step()
        # Sanity check: give the correct goal state:
        goal_states = self.expand_np_to_var(self.env.multitask_goal)
        new_obs = torch.cat(
            (
                obs,
                goal_states,
                taus,
            ),
            dim=1,
        )
        actions = self.policy(new_obs)
        return ptu.get_numpy(goal_states), ptu.get_numpy(actions)

    def get_action(self, ob):
        obs = self.expand_np(ob)
        # if self.policy is None:
        if True:
            actions, goal_states = self.get_feasible_actions_and_goal_states(
                ob
            )
        else:
            goal_states, actions = (
                self.get_feasible_goal_states_and_tdm_actions(ob)
            )
        env_cost = self.env.cost_fn(obs, actions, goal_states)
        env_cost = np.expand_dims(env_cost, 1)
        taus = self.expand_np_to_var(np.array([0]))
        feasibility_cost = - (
            self.tdm.eval_np(obs, goal_states, taus, actions)
        )
        if len(feasibility_cost.shape) > 1:
            feasibility_cost = feasibility_cost.sum(axis=1)[:, None]
        print(
            "weighted feasibility_cost",
            feasibility_cost.mean() * self.feasibility_weight,
        )
        print("env_Cost", env_cost.mean())
        costs = env_cost + feasibility_cost * self.feasibility_weight
        min_i = np.argmin(costs)
        return actions[min_i, :], {}


class SlsqpCMC(UniversalPolicy, nn.Module):
    """
    CMC = Collocation MPC Controller

    Implement

        pi(s_1, g) = pi_{distance}(s_1, s_2)

    where pi_{distance} is the SDQL policy and

        s_2 = argmin_{s_2} min_{s_{3:T+1}} ||s_{T+1} - g||_2^2
        subject to C(s_i, pi_{distance}(s_i, s_{i+1}), s_{i+1}) = 0

    for i = 1, ..., T, where C is an implicit model.

    using SLSQP
    """
    def __init__(
            self,
            implicit_model,
            env,
            solver_params=None,
            planning_horizon=1,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.implicit_model = implicit_model
        self.env = env
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.solver_params = solver_params
        self.planning_horizon = planning_horizon

        self.last_solution = None
        self.lower_bounds = np.hstack((
            self.env.action_space.low,
            self.env.observation_space.low
        ))
        self.upper_bounds = np.hstack((
            self.env.action_space.high,
            self.env.observation_space.high
        ))
        # TODO(vitchyr): figure out what to do if the state bounds are infinity
        # self.lower_bounds = - np.ones_like(self.lower_bounds)
        # self.upper_bounds = np.ones_like(self.upper_bounds)
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.constraints = {
            'type': 'eq',
            'fun': self.constraint_fctn,
            'jac': self.constraint_jacobian,
        }

    def split(self, x):
        """
        split into action, next_state
        """
        return x[:self.action_dim], x[self.action_dim:]

    def cost_function(self, x):
        action, next_state = self.split(x)
        return self.env.cost_fn(None, action, next_state)

    def cost_jacobian(self, x):
        jacobian = np.zeros_like(x)
        _, next_state = self.split(x)
        # TODO(vitchyr): stop hardcoding this
        jacobian[2:4] = (
            2 * (self.env.convert_ob_to_goal(next_state) - self.env.multitask_goal)
        )
        return jacobian

    def _constraint_fctn(self, x, state, return_grad):
        state = ptu.np_to_var(state)
        x = ptu.np_to_var(x, requires_grad=return_grad)
        action, next_state = self.split(x)
        action = action[None]
        next_state = next_state[None]

        state = state.unsqueeze(0)
        loss = self.implicit_model(state, action, next_state)
        if return_grad:
            loss.squeeze(0).backward()
            return ptu.get_numpy(x.grad)
        else:
            return ptu.get_numpy(loss.squeeze(0))[0]

    def constraint_fctn(self, x, state=None):
        return self._constraint_fctn(x, state, False)

    def constraint_jacobian(self, x, state=None):
        return self._constraint_fctn(x, state, True)

    def reset(self):
        self.last_solution = None

    def get_action(self, obs):
        if self.last_solution is None:
            self.last_solution = np.hstack((
                np.zeros(self.action_dim),
                np.tile(obs, self.planning_horizon),
            ))
        self.constraints['args'] = (obs, )
        result = optimize.minimize(
            self.cost_function,
            self.last_solution,
            jac=self.cost_jacobian,
            constraints=self.constraints,
            method='SLSQP',
            options=self.solver_params,
            bounds=self.bounds,
        )
        if not result.success:
            print("WARNING: SLSQP Did not succeed. Message is:")
            print(result.message)

        action, _ = self.split(result.x)
        self.last_solution = result.x
        return action, {}
