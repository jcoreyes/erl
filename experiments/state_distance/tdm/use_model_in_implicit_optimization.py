import argparse
import joblib
import numpy as np
from scipy import optimize

import torch
import torch.nn as nn
from railrl.core import logger
from railrl.samplers.util import rollout
from railrl.state_distance.policies import UniversalPolicy
from railrl.state_distance.probabilistic_tdm.mpc_controller import \
    ImplicitMPCController
import railrl.torch.pytorch_util as ptu
from railrl.torch.core import PyTorchModule

# PATH = '/home/vitchyr/git/railrl/data/local/01-19-reacher-model-based/01-19' \
#        '-reacher-model-based_2018_01_19_15_54_27_0000--s-983077/params.pkl'
PATH = '/home/vitchyr/git/railrl/data/local/01-27-reacher-full-mpcnn-H1/01-27-reacher-full-mpcnn-H1_2018_01_27_17_59_04_0000--s-96642/params.pkl'


class ModelToTdm(PyTorchModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, goal, taus, action):
        out = (goal - obs - self.model(obs, action))**2
        return -torch.norm(out, dim=1).unsqueeze(1)


# class StateOnlySdqBasedSqpOcPolicy(UniversalPolicy, nn.Module):
class SqpOcPolicy(UniversalPolicy, nn.Module):
    """
    Implement

        pi(s_1, g) = pi_{distance}(s_1, s_2)

    where pi_{distance} is the SDQL policy and

        s_2 = argmin_{s_2} min_{s_{3:T+1}} ||s_{T+1} - g||_2^2
        subject to Q(s_i, pi_{distance}(s_i, s_{i+1}), s_{i+1}) = 0

    for i = 1, ..., T

    using SLSQP
    """
    def __init__(
            self,
            qf,
            env,
            solver_params=None,
            planning_horizon=1,
    ):
        super().__init__()
        nn.Module.__init__(self)
        self.qf = qf
        self.env = env
        self.action_dim = self.env.action_space.low.size
        self.obs_dim = self.env.observation_space.low.size
        self.solver_params = solver_params
        self.planning_horizon = planning_horizon

        self.last_solution = None
        self.lower_bounds = np.hstack((
            -np.ones(self.action_dim),
            self.env.observation_space.low
        ))
        self.upper_bounds = np.hstack((
            np.ones(self.action_dim),
            self.env.observation_space.high
        ))
        # TODO(vitchyr): figure out what to do if the state bounds are infinity
        self.lower_bounds = - np.ones_like(self.lower_bounds)
        self.upper_bounds = np.ones_like(self.upper_bounds)
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
        import ipdb; ipdb.set_trace()
        jacobian = np.zeros_like(x)
        all_next_states = self.split(x)
        last_state = all_next_states[-1, :]
        # Assuming the last `self.observation_dim` part of x is the last state
        jacobian[-self.observation_dim:] = (
                2 * (last_state - self._goal_np)
        )
        return jacobian

    def _constraint_fctn(self, x, state, return_grad):
        state = ptu.np_to_var(state)
        x = ptu.np_to_var(x, requires_grad=return_grad)
        action, next_state = self.split(x)

        state = state.unsqueeze(0)
        import ipdb; ipdb.set_trace()
        loss = - self.qf(
            state, action, next_state, self._tau_expanded_torch
        )
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
        next_goal_state = result.x[:self.observation_dim]
        action = self.get_np_action(obs, next_goal_state)
        if np.isnan(action).any():
            logger.log("WARNING: SLSQP returned nan. Adding noise to last "
                       "action")
            action = self.last_solution[:self.action_dim] + np.random.uniform(
                self.env.action_space.low,
                self.env.action_space.high,
            ) / 100
        else:
            self.last_solution = result.x
        return action, {}

    def get_np_action(self, state_np, goal_state_np):
        return ptu.get_numpy(
            self.policy(
                ptu.np_to_var(np.expand_dims(state_np, 0)),
                ptu.np_to_var(np.expand_dims(goal_state_np, 0)),
                self._tau_expanded_torch,
            ).squeeze(0)
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        default=PATH,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--mtau', type=float,
                        help='Max tau value')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--dt', help='decrement tau', action='store_true')
    parser.add_argument('--cycle', help='cycle tau', action='store_true')
    parser.add_argument('--ndc', help='not (decrement and cycle tau)',
                        action='store_true')
    parser.add_argument('--justsim', action='store_true')
    parser.add_argument('--npath', type=int, default=100)
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    data = joblib.load(args.file)
    env = data['env']
    num_samples = 1000
    resolution = 10
    if 'policy' in data:
        trained_mpc_controller = data['policy']
    else:
        trained_mpc_controller = data['exploration_policy'].policy
    trained_mpc_controller.env = env
    trained_mpc_controller.cost_fn = env.cost_fn
    trained_mpc_controller.num_simulated_paths = args.npath
    trained_mpc_controller.horizon = 1
    if args.justsim:
        while True:
            path = rollout(
                env,
                trained_mpc_controller,
                max_path_length=args.H,
                animated=not args.hide,
            )
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics([path])
            logger.dump_tabular()
    else:

        model = data['model']
        tdm = ModelToTdm(model)

        for weight in [100]:
            for num_simulated_paths in [args.npath]:
                print("")
                print("weight", weight)
                print("num_simulated_paths", num_simulated_paths)
                # policy = ImplicitMPCController(
                #     env,
                #     tdm,
                #     None,
                #     num_simulated_paths=num_simulated_paths,
                #     feasibility_weight=weight,
                # )
                # policy = trained_mpc_controller
                policy = SqpOcPolicy(tdm, env)
                policy.train(False)
                paths = []
                for _ in range(5):
                    paths.append(rollout(
                        env,
                        policy,
                        max_path_length=args.H,
                        animated=not args.hide,
                    ))
                if hasattr(env, "log_diagnostics"):
                    env.log_diagnostics(paths)
                final_distance = logger.get_table_dict()['Final Euclidean distance to goal Mean']
                print("final distance", final_distance)
                # logger.dump_tabular()
