import time

import numpy as np
import torch
from scipy import optimize

from railrl.state_distance.policies import UniversalPolicy
from railrl.torch import pytorch_util as ptu


class BetaLbfgsController(UniversalPolicy):
    """
    Solve

        min_{s_1:T} \sum_t c(s_t) beta(s_t | s_{t-1})

    using L-BFGS-boxed where

        c(s_t) = ||s_t - goal||
        beta(a, b) = prob(reach a | at state b)

    """
    def __init__(
            self,
            beta,
            env,
            goal_slice,
            multitask_goal_slice,
            max_cost,
            planning_horizon=1,
            warm_start=False,
            solver_kwargs=None,
            only_use_terminal_env_loss=False,
            replan_every_time_step=True,
            learned_policy=None,
            use_learned_policy=False,
    ):
        super().__init__()
        if solver_kwargs is None:
            solver_kwargs = {}
        self.beta = beta
        self.env = env
        self.goal_slice = goal_slice
        self.multitask_goal_slice = multitask_goal_slice
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.ao_dim = self.action_dim + self.obs_dim
        self.planning_horizon = planning_horizon
        self.warm_start = warm_start
        self.solver_kwargs = solver_kwargs
        self.only_use_terminal_env_loss = only_use_terminal_env_loss
        self.replan_every_time_step = replan_every_time_step
        self.t_in_plan = 0
        self.learned_policy = learned_policy
        self.min_lm = 0.1
        self.max_lm = 1000
        self.error_threshold = 0.5
        self.num_steps_left = ptu.np_to_var(
            np.zeros((self.planning_horizon, 1))
        )
        self.max_cost = max_cost
        self.use_learned_policy = use_learned_policy

        self.last_solution = None
        self.best_action_seq = None
        self.best_obs_seq = None
        self.desired_features_torch = None
        self.totals = []
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
        self.bounds = list(zip(self.lower_bounds, self.upper_bounds))
        self.forward = 0
        self.backward = 0

    def batchify(self, x, current_ob):
        """
        Convert
            [a1, s2, a2, s3, a3, s4]
        into
            [s1, s2, s3], [a1, a2, a3], [s2, s3, s4]
        """
        obs = []
        actions = []
        next_obs = []
        ob = current_ob
        for h in range(self.planning_horizon):
            start_h = h * self.ao_dim
            next_ob = x[start_h+self.action_dim:start_h+self.ao_dim]
            obs.append(ob)
            actions.append(x[start_h:start_h+self.action_dim])
            next_obs.append(next_ob)
            ob = next_ob
        return (
            torch.stack(obs),
            torch.stack(actions),
            torch.stack(next_obs),
        )

    def _env_cost_function(self, x, current_ob):
        _, _, next_obs = self.batchify(x, current_ob)
        next_features_predicted = next_obs[:, self.goal_slice]
        if self.only_use_terminal_env_loss:
            diff = (
                    next_features_predicted[-1] - self.desired_features_torch[-1]
            )
            loss = (diff**2).sum()
        else:
            diff = next_features_predicted - self.desired_features_torch
            loss = (diff**2).sum()
        return loss

    def _feasibility_probabilities(self, x, current_ob):
        obs, actions, next_obs = self.batchify(x, current_ob)
        # TODO: computing cumulative product of probabilities
        return self.beta(obs, actions, next_obs, self.num_steps_left)

    def cost_function(self, x, current_ob):
        self.forward -= time.time()
        x = ptu.np_to_var(x, requires_grad=True)
        current_ob = ptu.np_to_var(current_ob)
        env_costs = self._env_cost_function(x, current_ob)
        probabilities = self._feasibility_probabilities(x, current_ob)
        loss = env_costs * probabilities + (1-probabilities) * self.max_cost
        loss = loss.sum()
        loss_np = ptu.get_numpy(loss)[0].astype(np.float64)
        self.forward += time.time()
        self.backward -= time.time()
        loss.backward()
        gradient_np = ptu.get_numpy(x.grad).astype(np.float64)
        self.backward += time.time()
        return loss_np, gradient_np

    def reset(self):
        self.last_solution = None

    def get_action(self, current_ob):
        goal = self.env.multitask_goal[self.multitask_goal_slice]
        return self._get_action(current_ob, goal)

    def _get_action(self, current_ob, goal):
        if (
                self.replan_every_time_step
                or self.t_in_plan == self.planning_horizon
                or self.last_solution is None
        ):
            full_solution = self.replan(current_ob, goal)

            x_torch = ptu.np_to_var(full_solution, requires_grad=True)
            current_ob_torch = ptu.np_to_var(current_ob)

            _, actions, next_obs = self.batchify(x_torch, current_ob_torch)
            self.best_action_seq = np.array([ptu.get_numpy(a) for a in actions])
            self.best_obs_seq = np.array(
                [current_ob] + [ptu.get_numpy(o) for o in next_obs]
            )

            self.last_solution = full_solution
            self.t_in_plan = 0

        learned_actions = self.learned_policy.eval_np(
            self.best_obs_seq[:-1],
            self.best_obs_seq[1:],
            np.zeros((self.planning_horizon, 1))
        )
        if self.use_learned_policy:
            best_action_seq = learned_actions
        else:
            best_action_seq = self.best_action_seq[self.t_in_plan:]
        agent_info = dict(
            best_action_seq=best_action_seq,
            lbfgs_action_seq=self.best_action_seq[self.t_in_plan:],
            learned_action_seq=learned_actions,
            best_obs_seq=self.best_obs_seq[self.t_in_plan:],
        )
        action = best_action_seq[0]
        self.t_in_plan += 1

        return action, agent_info

    def replan(self, current_ob, goal):
        if self.last_solution is None or not self.warm_start:
            solution = []
            for i in range(self.planning_horizon):
                solution.append(self.env.action_space.sample())
                solution.append(current_ob)
            self.last_solution = np.hstack(solution)
        self.desired_features_torch = ptu.np_to_var(
            goal[None].repeat(self.planning_horizon, 0)
        )
        self.forward = self.backward = 0
        start = time.time()
        x, f, d = optimize.fmin_l_bfgs_b(
            self.cost_function,
            self.last_solution,
            args=(current_ob,),
            bounds=self.bounds,
            **self.solver_kwargs
        )
        total = time.time() - start
        self.totals.append(total)
        warnflag = d['warnflag']
        if warnflag != 0:
            if warnflag == 1:
                print("too many function evaluations or too many iterations")
            else:
                print(d['task'])
        return x


class BetaLBfgsBCMC(BetaLbfgsController):
    """
    Basically the same as LBfgsBCMC but use the goal passed into get_action

    TODO: maybe use num_steps_left to replace t_in_plan?
    """
    def get_action(self, current_ob, goal, num_steps_left):
        return self._get_action(current_ob, goal)