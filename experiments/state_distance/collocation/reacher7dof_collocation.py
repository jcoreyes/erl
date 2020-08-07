import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from rlkit.samplers.util import rollout
from rlkit.state_distance.util import merge_into_flat_obs
from rlkit.torch.core import PyTorchModule
from rlkit.torch.mpc.collocation.collocation_mpc_controller import SlsqpCMC, \
    GradientCMC, StateGCMC, LBfgsBCMC, Reacher7DofLBfgsBCMC
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger

# Reacher7DofFullGoal - TDM
from rlkit.torch.mpc.collocation.model_to_implicit_model import \
    ModelToImplicitModel

PATH = '/home/vitchyr/git/railrl/data/doodads3/02-07-reacher7dof-sac-mtau-1-or-10-terminal-bonus/02-07-reacher7dof-sac-mtau-1-or-10-terminal-bonus-id4-s9821/params.pkl'
GOAL_SLICE = slice(0, 7)
MULTITASK_GOAL_SLICE = GOAL_SLICE


# Reacher7DofFullGoal - Model
PATH = '/home/vitchyr/git/railrl/data/local/01-27-reacher-full-mpcnn-H1/01-27-reacher-full-mpcnn-H1_2018_01_27_17_59_04_0000--s-96642/params.pkl'
# GOAL_SLICE = slice(0, 7)
GOAL_SLICE = slice(14, 17)
MULTITASK_GOAL_SLICE = GOAL_SLICE


class TdmToImplicitModel(PyTorchModule):
    def __init__(self, env, qf, tau, output_scale=1):
        super().__init__()
        self.env = env
        self.qf = qf
        self.tau = tau
        self.output_scale = output_scale

    def forward(self, states, actions, next_states):
        taus = ptu.np_to_var(
            self.tau * np.ones((states.shape[0], 1))
        )
        goals = self.env.convert_obs_to_goals(next_states)
        flat_obs = merge_into_flat_obs(states, goals, taus)
        return self.qf(flat_obs, actions).sum(1) * self.output_scale


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

    data = joblib.load(args.file)
    env = data['env']

    # variant_path = Path(args.file).parents[0] / 'variant.json'
    # variant = json.load(variant_path.open())
    # reward_scale = variant['sac_tdm_kwargs']['base_kwargs']['reward_scale']
    # qf = data['qf']
    # implicit_model = TdmToImplicitModel(
    #     env,
    #     qf,
    #     tau=0,
    #     output_scale=1./reward_scale,
    # )

    model = data['model']
    implicit_model = ModelToImplicitModel(
        model,
        order=2,
    )

    lagrange_multiplier = 100
    planning_horizon = 1
    optimizer = args.opt
    print("Optimizer choice: ", optimizer)
    policy = Reacher7DofLBfgsBCMC(
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
        # env.set_goal(env.sample_goal_for_rollout())
        paths.append(rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        ))
        env.log_diagnostics(paths)
        logger.dump_tabular()
