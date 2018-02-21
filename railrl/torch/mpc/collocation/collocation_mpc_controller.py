import numpy as np
import torch
from scipy import optimize
from torch import optim, nn as nn

from railrl.state_distance.policies import UniversalPolicy
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule


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
            goal_slice,
            multitask_goal_slice,
            solver_params=None,
            planning_horizon=1,
            use_implicit_model_gradient=False,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.implicit_model = implicit_model
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.solver_params = solver_params
        self.use_implicit_model_gradient = use_implicit_model_gradient
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
        self.lower_bounds = np.tile(self.lower_bounds, self.planning_horizon)
        self.upper_bounds = np.tile(self.upper_bounds, self.planning_horizon)
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
        actions_and_obs = []
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            actions_and_obs.append((
                x[start_h:start_h+self.action_dim],
                x[start_h+self.action_dim:start_h+self.ao_dim],
            ))
        return actions_and_obs

    def _cost_function(self, x, order):
        x = ptu.np_to_var(x, requires_grad=True)
        loss = 0
        for action, next_state in self.split(x):
            next_features_predicted = next_state[self.goal_slice]
            desired_features = ptu.np_to_var(
                self.env.multitask_goal[self.multitask_goal_slice]
                * np.ones(next_features_predicted.shape)
            )
            diff = next_features_predicted - desired_features
            loss += (diff**2).sum()
        if order == 0:
            return ptu.get_numpy(loss)[0]
        elif order == 1:
            loss.squeeze(0).backward()
            return ptu.get_numpy(x.grad)

    def cost_function(self, x):
        return self._cost_function(x, order=0)
        # action, next_state = self.split(x)
        # return self.env.cost_fn(None, action, next_state)

    def cost_jacobian(self, x):
        return self._cost_function(x, order=1)
        # jacobian = np.zeros_like(x)
        # _, next_state = self.split(x)
        # full_gradient = (
        #         2 * (self.env.convert_ob_to_goal(next_state) - self.env.multitask_goal)
        # )
        # jacobian[7:14] = full_gradient[:7]
        # return jacobian

    def _constraint_fctn(self, x, state, order):
        state = ptu.np_to_var(state)
        state = state.unsqueeze(0)
        x = ptu.np_to_var(x, requires_grad=order > 0)
        loss = 0
        for action, next_state in self.split(x):
            action = action[None]
            next_state = next_state[None]

            loss += self.implicit_model(state, action, next_state)
            state = next_state
        if order == 0:
            return ptu.get_numpy(loss.squeeze(0))[0]
        elif order == 1:
            loss.squeeze(0).backward()
            return ptu.get_numpy(x.grad)
        else:
            grad_params = torch.autograd.grad(loss, x, create_graph=True)[0]
            grad_norm = torch.dot(grad_params, grad_params)
            grad_norm.backward()
            return ptu.get_numpy(x.grad)

    def constraint_fctn(self, x, state=None):
        if self.use_implicit_model_gradient:
            grad = self._constraint_fctn(x, state, 1)
            return np.inner(grad, grad)
        else:
            return self._constraint_fctn(x, state, 0)

    def constraint_jacobian(self, x, state=None):
        if self.use_implicit_model_gradient:
            return self._constraint_fctn(x, state, 2)
        else:
            return self._constraint_fctn(x, state, 1)

    def reset(self):
        self.last_solution = None

    def get_action(self, obs):
        if self.last_solution is None:
            init_solution = np.hstack((
                np.zeros(self.action_dim),
                obs,
            ))
            self.last_solution = np.tile(init_solution, self.planning_horizon)
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

        action, _ = self.split(result.x)[0]
        self.last_solution = result.x
        return action, {}


class GradientCMC(UniversalPolicy, nn.Module):
    """
    CMC = Collocation MPC Controller

    Implement

        pi(s_1, g) = pi_{distance}(s_1, s_2)

    where pi_{distance} is the SDQL policy and

        s_2 = argmin_{s_2} min_{s_{3:T+1}} ||s_{T+1} - g||_2^2
        subject to C(s_i, pi_{distance}(s_i, s_{i+1}), s_{i+1}) = 0

    for i = 1, ..., T, where C is an implicit model.

    using gradient descent.

    Each element of "x" through the code represents the vector
    [a_1, s_1, a_2, s_2, ..., a_T, s_T]
    """
    def __init__(
            self,
            implicit_model,
            env,
            goal_slice,
            multitask_goal_slice,
            lagrange_multiplier=1,
            num_particles=1,
            num_grad_steps=10,
            learning_rate=1e-1,
            warm_start=False,
            planning_horizon=1,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.implicit_model = implicit_model
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.lagrange_multiplier = lagrange_multiplier
        self.planning_horizon = planning_horizon
        self.num_particles = num_particles
        self.num_grad_steps = num_grad_steps
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.last_solution = None

    def split(self, x):
        """
        split into action, next_state
        """
        actions_and_obs = []
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            actions_and_obs.append((
                x[:, start_h:start_h+self.action_dim],
                x[:, start_h+self.action_dim:start_h+self.ao_dim],
            ))
        return actions_and_obs

    def _expand_np_to_var(self, array, requires_grad=False):
        array_expanded = np.repeat(
            np.expand_dims(array, 0),
            self.num_particles,
            axis=0
        )
        return ptu.np_to_var(array_expanded, requires_grad=requires_grad)

    def cost_function(self, state, actions, next_states):
        """
        :param x: a PyTorch Variable.
        :return:
        """
        loss = 0
        for i in range(self.planning_horizon):
            slc = slice(i*self.obs_dim, (i+1)*self.obs_dim)
            next_state = next_states[:, slc]
            next_features_predicted = next_state[:, self.goal_slice]
            desired_features = ptu.np_to_var(
                self.env.multitask_goal[self.multitask_goal_slice][None]
                * np.ones(next_features_predicted.shape)
            )
            diff = next_features_predicted - desired_features
            loss += (diff**2).sum(dim=1, keepdim=True)
        return loss

    def constraint_fctn(self, state, actions, next_states):
        """
        :param x: a PyTorch Variable.
        :param state: a PyTorch Variable.
        :return:
        """
        loss = 0
        for i in range(self.planning_horizon):
            next_state = next_states[:, i*self.obs_dim:(i+1)*self.obs_dim]
            action = actions[:, i*self.action_dim:(i+1)*self.action_dim]

            loss -= self.implicit_model(state, action, next_state)
            state = next_state
        return loss

    def sample_actions(self):
        return np.random.uniform(
            self.action_low,
            self.action_high,
            (self.num_particles, self.action_dim)
        )

    def get_action(self, ob):
        if self.last_solution is None or not self.warm_start:
            init_solution = []
            for _ in range(self.planning_horizon):
                init_solution.append(self.sample_actions())
            for _ in range(self.planning_horizon):
                init_solution.append(
                    np.repeat(ob[None], self.num_particles, axis=0)
                )

            self.last_solution = np.hstack(init_solution)

        ob = self._expand_np_to_var(ob)
        x = ptu.np_to_var(self.last_solution, requires_grad=True)

        optimizer = optim.Adam([x], lr=self.learning_rate)
        loss = None
        for i in range(self.num_grad_steps):
            actions = x[:, :self.action_dim * self.planning_horizon]
            actions = torch.clamp(actions, -1, 1)
            next_states = x[:, self.action_dim * self.planning_horizon:]
            loss = (
                self.cost_function(ob, actions, next_states)
                + self.lagrange_multiplier *
                    self.constraint_fctn(ob, actions, next_states)
            )
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            x[:, :self.action_dim * self.planning_horizon].data = torch.clamp(
                x[:, :self.action_dim * self.planning_horizon].data, -1, 1
            )

        self.last_solution = ptu.get_numpy(x)
        if loss is None:
            actions = x[:, :self.action_dim * self.planning_horizon]
            actions = torch.clamp(actions, -1, 1)
            next_states = x[:, self.action_dim * self.planning_horizon:]
            loss = (
                    self.cost_function(ob, actions, next_states)
                    + self.lagrange_multiplier *
                    self.constraint_fctn(ob, actions, next_states)
            )
        loss_np = ptu.get_numpy(loss).sum(axis=1)
        min_i = np.argmin(loss_np)
        action = self.last_solution[min_i, :self.action_dim]
        action = np.clip(action, -1, 1)
        return action, {}


class StateGCMC(GradientCMC):
    """
    Use gradient-based optimization for choosing the next state, but using
    stochastic optimization for choosing the action.
    """
    def get_action(self, ob):
        if self.last_solution is None or not self.warm_start:
            init_solution = []
            for _ in range(self.planning_horizon):
                init_solution.append(
                    np.repeat(ob[None], self.num_particles, axis=0)
                )

            self.last_solution = np.hstack(init_solution)

        ob = self._expand_np_to_var(ob)
        actions_np = np.hstack(
            [self.sample_actions() for _ in range(self.planning_horizon)]
        )
        actions = ptu.np_to_var(actions_np)
        next_states = ptu.np_to_var(self.last_solution, requires_grad=True)

        optimizer = optim.Adam([next_states], lr=self.learning_rate)
        for i in range(self.num_grad_steps):
            constraint_loss = self.constraint_fctn(ob, actions, next_states)
            optimizer.zero_grad()
            constraint_loss.sum().backward()
            optimizer.step()

        final_loss = (
            self.cost_function(ob, actions, next_states)
            + self.lagrange_multiplier *
            self.constraint_fctn(ob, actions, next_states)
        )
        self.last_solution = ptu.get_numpy(next_states)
        final_loss_np = ptu.get_numpy(final_loss).sum(axis=1)
        min_i = np.argmin(final_loss_np)
        action = actions_np[min_i, :self.action_dim]
        return action, {}

import matplotlib.pyplot as plt


class LBfgsBCMC(UniversalPolicy):
    def __init__(
            self,
            implicit_model,
            env,
            goal_slice,
            multitask_goal_slice,
            planning_horizon=1,
            lagrange_multipler=1,
            warm_start=False,
            solver_params=None,
    ):
        super().__init__()
        if solver_params is None:
            solver_params = {}
        self.implicit_model = implicit_model
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.planning_horizon = planning_horizon
        self.lagrange_multipler = lagrange_multipler
        self.warm_start = warm_start
        self.solver_params = solver_params

        self.last_solution = None
        self.lower_bounds = np.hstack((
            self.env.action_space.low,
            self.env.observation_space.low
        ))
        self.upper_bounds = np.hstack((
            self.env.action_space.high,
            self.env.observation_space.high
        ))
        self.lower_bounds = np.tile(self.lower_bounds, self.planning_horizon)
        self.upper_bounds = np.tile(self.upper_bounds, self.planning_horizon)
        # TODO(vitchyr): figure out what to do if the state bounds are infinity
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        fig, (self.ax1, self.ax2) = plt.subplots(1, 2)

    def split(self, x):
        """
        split into action, next_state
        """
        actions_and_obs = []
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            actions_and_obs.append((
                x[start_h:start_h+self.action_dim],
                x[start_h+self.action_dim:start_h+self.ao_dim],
            ))
        return actions_and_obs

    def _env_cost_function(self, x):
        loss = 0
        for action, next_state in self.split(x):
            next_features_predicted = next_state[self.goal_slice]
            desired_features = ptu.np_to_var(
                self.env.multitask_goal[self.multitask_goal_slice]
                * np.ones(next_features_predicted.shape)
            )
            diff = next_features_predicted - desired_features
            # loss = loss + (diff**2).sum()
            loss = (diff**2).sum()  # Just use loss at last time step
        return loss

    def _feasibility_cost_function(self, x, state):
        state = ptu.np_to_var(state)
        state = state.unsqueeze(0)
        loss = 0
        for action, next_state in self.split(x):
            action = action[None]
            next_state = next_state[None]

            loss -= self.implicit_model(state, action, next_state)
            state = next_state
        return loss

    def cost_function(self, x, observation):
        x = ptu.np_to_var(x, requires_grad=True)
        loss = (
                self.lagrange_multipler
                * self._feasibility_cost_function(x, observation)
                + self._env_cost_function(x)
        )
        loss_np = ptu.get_numpy(loss)[0].astype(np.float64)
        loss.squeeze(0).backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        return loss_np, gradient_np

    def reset(self):
        self.last_solution = None

    def get_action(self, obs):
        if self.last_solution is None or not self.warm_start:
            init_solution = np.hstack((
                np.zeros(self.action_dim),
                obs,
            ))
            self.last_solution = np.tile(init_solution, self.planning_horizon)
            solution = []
            for i in range(self.planning_horizon):
                solution.append(self.env.action_space.sample())
                progress = (i + 1.) / self.planning_horizon
                avg = (
                  progress * self.env.multitask_goal
                  + (1-progress) * obs
                )
                solution.append(avg)
            # self.last_solution = np.hstack([
            #     [self.env.action_space.sample(), obs]
            #     for _ in range(self.planning_horizon)
            # ])
            self.last_solution = np.hstack(solution)
        x, f, d = optimize.fmin_l_bfgs_b(
            self.cost_function,
            self.last_solution,
            args=(obs,),
            bounds=self.bounds,
            **self.solver_params
        )
        warnflag = d['warnflag']
        if warnflag != 0:
            if warnflag == 1:
                print("too many function evaluations or too many iterations")
            else:
                print(d['task'])

        actions_and_obs = self.split(x)

        # For Point2d u-shaped wall
        # best_action_seq = np.array([a for a, o in actions_and_obs])
        # best_obs_seq = np.array([obs] + [o for a, o in actions_and_obs])
        # real_obs_seq = self.env.wrapped_env.wrapped_env.true_states(
        #     obs, best_action_seq
        # )
        # print("best_action_seq",best_action_seq)
        # self.ax1.clear()
        # self.env.wrapped_env.wrapped_env.plot_trajectory(
        #     self.ax1,
        #     np.array(best_obs_seq),
        #     np.array(best_action_seq),
        #     goal=self.env.wrapped_env.wrapped_env._target_position,
        # )
        # self.ax1.set_title("imagined")
        # self.ax2.clear()
        # self.env.wrapped_env.wrapped_env.plot_trajectory(
        #     self.ax2,
        #     np.array(real_obs_seq),
        #     np.array(best_action_seq),
        #     goal=self.env.wrapped_env.wrapped_env._target_position,
        # )
        # self.ax2.set_title("real")
        # plt.draw()
        # plt.pause(0.001)

        action = actions_and_obs[0][0]
        self.last_solution = x
        return action, {}


class TdmLBfgsBCMC(UniversalPolicy):
    def __init__(
            self,
            implicit_model,
            env,
            goal_slice,
            multitask_goal_slice,
            planning_horizon=1,
            lagrange_multipler=1,
            warm_start=False,
            solver_kwargs=None,
    ):
        super().__init__()
        if solver_kwargs is None:
            solver_kwargs = {}
        self.implicit_model = implicit_model
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.planning_horizon = planning_horizon
        self.lagrange_multipler = lagrange_multipler
        self.warm_start = warm_start
        self.solver_kwargs = solver_kwargs

        self.last_solution = None
        self.lower_bounds = np.hstack((
            self.env.action_space.low,
            self.env.observation_space.low
        ))
        self.upper_bounds = np.hstack((
            self.env.action_space.high,
            self.env.observation_space.high
        ))
        self.lower_bounds = np.tile(self.lower_bounds, self.planning_horizon)
        self.upper_bounds = np.tile(self.upper_bounds, self.planning_horizon)
        # TODO(vitchyr): figure out what to do if the state bounds are infinity
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))

    def split(self, x):
        """
        split into action, next_state
        """
        actions_and_obs = []
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            actions_and_obs.append((
                x[start_h:start_h+self.action_dim],
                x[start_h+self.action_dim:start_h+self.ao_dim],
            ))
        return actions_and_obs

    def _env_cost_function(self, x, goal):
        loss = 0
        for action, next_state in self.split(x):
            next_features_predicted = next_state[self.goal_slice]
            diff = next_features_predicted - goal[self.multitask_goal_slice]
            loss += (diff**2).sum()
        return loss

    def _feasibility_cost_function(self, x, state):
        state = ptu.np_to_var(state)
        state = state.unsqueeze(0)
        loss = 0
        for action, next_state in self.split(x):
            action = action[None]
            next_state = next_state[None]

            loss -= self.implicit_model(state, action, next_state)
            state = next_state
        return loss

    def cost_function(self, x, observation, goal):
        x = ptu.np_to_var(x, requires_grad=True)
        goal = ptu.np_to_var(goal)
        loss = (
                self.lagrange_multipler
                * self._feasibility_cost_function(x, observation)
                + self._env_cost_function(x, goal)
        )
        loss_np = ptu.get_numpy(loss)[0].astype(np.float64)
        loss.squeeze(0).backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        return loss_np, gradient_np

    def reset(self):
        self.last_solution = None

    def get_action(self, ob, goal, num_steps_left):
        if self.last_solution is None or not self.warm_start:
            init_solution = np.hstack((
                np.zeros(self.action_dim),
                ob,
            ))
            self.last_solution = np.tile(init_solution, self.planning_horizon)
        x, f, d = optimize.fmin_l_bfgs_b(
            self.cost_function,
            self.last_solution,
            args=(ob, goal),
            bounds=self.bounds,
            **self.solver_kwargs
        )
        warnflag = d['warnflag']
        if warnflag != 0:
            if warnflag == 1:
                print("too many function evaluations or too many iterations")
            else:
                print(d['task'])

        action, _ = self.split(x)[0]
        self.last_solution = x
        return action, {}


class TdmToImplicitModel(PyTorchModule):
    def __init__(self, env, qf, tau):
        self.quick_init(locals())
        super().__init__()
        self.env = env
        self.qf = qf
        self.tau = tau

    def forward(self, states, actions, next_states):
        taus = ptu.np_to_var(
            self.tau * np.ones((states.shape[0], 1))
        )
        goals = self.env.convert_obs_to_goals(next_states)
        return self.qf(
            observations=states,
            actions=actions,
            goals=goals,
            num_steps_left=taus,
        ).sum(1)
