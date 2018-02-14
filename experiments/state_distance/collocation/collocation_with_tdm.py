import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from railrl.samplers.util import rollout
from railrl.state_distance.util import merge_into_flat_obs
from railrl.torch.core import PyTorchModule
from railrl.torch.mpc.collocation.collocation_mpc_controller import SlsqpCMC, \
    GradientCMC, StateGCMC, LBfgsBCMC
import railrl.torch.pytorch_util as ptu
from railrl.core import logger

# Reacher7DofFullGoal - TDM
PATH = '/home/vitchyr/git/railrl/data/doodads3/02-07-reacher7dof-sac-mtau-1-or-10-terminal-bonus/02-07-reacher7dof-sac-mtau-1-or-10-terminal-bonus-id4-s9821/params.pkl'
GOAL_SLICE = slice(0, 7)

# point2d - TDM
# PATH = 'data/local/02-08-dev-sac-tdm-launch/02-08-dev-sac-tdm-launch_2018_02_08_22_50_27_0000--s-5908/params.pkl'
# GOAL_SLICE = slice(0, 2)

MULTITASK_GOAL_SLICE = GOAL_SLICE

# Reacher7DofXyzGoalState
# tau max = 1
# PATH = '/home/vitchyr/git/railrl/data/doodads3/02-08-reacher7dof-3d-sac-mtau-0-1-or-10-terminal-bonus/02-08-reacher7dof-3d-sac-mtau-0-1-or-10-terminal-bonus-id7-s327/params.pkl'
# tau max = 0
# PATH = '/home/vitchyr/git/railrl/data/doodads3/02-08-reacher7dof-sac-squared-distance-sweep-qf-activation-2/02-08-reacher7dof-sac-squared-distance-sweep-qf-activation-2-id1-s5793/params.pkl'
# GOAL_SLICE = slice(14, 17)
# MULTITASK_GOAL_SLICE = slice(0, 3)


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
        flat_obs = merge_into_flat_obs(states, goals, taus)
        return self.qf(flat_obs, actions).sum(1)


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
    parser.add_argument('--justsim', action='store_true')
    parser.add_argument('--npath', type=int, default=100)
    parser.add_argument('--opt', type=str, default='lbfgs')
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    variant_path = Path(args.file).parents[0] / 'variant.json'
    variant = json.load(variant_path.open())
    reward_scale = variant['sac_tdm_kwargs']['base_kwargs']['reward_scale']

    data = joblib.load(args.file)
    env = data['env']
    qf = data['qf']

    implicit_model = TdmToImplicitModel(
        env,
        qf,
        tau=0,
    )
    # lagrange_multiplier = 100 / reward_scale
    lagrange_multiplier = 10
    planning_horizon = 3
    optimizer = args.opt
    print("Optimizer choice: ", optimizer)
    if optimizer == 'slsqp':
        policy = SlsqpCMC(
            implicit_model,
            env,
            goal_slice=GOAL_SLICE,
            multitask_goal_slice=MULTITASK_GOAL_SLICE,
            planning_horizon=planning_horizon,
            # use_implicit_model_gradient=True,
            solver_params={
                'ftol': 1e-2,
                'maxiter': 100,
            },
        )
    elif optimizer == 'gradient':
        policy = GradientCMC(
            implicit_model,
            env,
            goal_slice=GOAL_SLICE,
            multitask_goal_slice=MULTITASK_GOAL_SLICE,
            planning_horizon=planning_horizon,
            lagrange_multiplier=lagrange_multiplier,
            num_grad_steps=100,
            num_particles=128,
            warm_start=False,
        )
    elif optimizer == 'state':
        policy = StateGCMC(
            implicit_model,
            env,
            goal_slice=GOAL_SLICE,
            multitask_goal_slice=MULTITASK_GOAL_SLICE,
            planning_horizon=planning_horizon,
            lagrange_multiplier=lagrange_multiplier,
            num_grad_steps=100,
            num_particles=128,
            warm_start=False,
        )
    elif optimizer == 'lbfgs':
        policy = LBfgsBCMC(
            implicit_model,
            env,
            goal_slice=GOAL_SLICE,
            multitask_goal_slice=MULTITASK_GOAL_SLICE,
            lagrange_multipler=lagrange_multiplier,
            planning_horizon=planning_horizon,
            solver_params={
                'factr': 1e9,
            },
        )
    paths = []
    while True:
        env.set_goal(env.sample_goal_for_rollout())
        paths.append(rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        ))
        env.log_diagnostics(paths)
        logger.dump_tabular()
