import argparse
import joblib

from rlkit.samplers.util import rollout
from rlkit.torch.mpc.collocation.collocation_mpc_controller import SlsqpCMC, \
    GradientCMC, StateGCMC, LBfgsBCMC
from rlkit.torch.mpc.collocation.model_to_implicit_model import \
    ModelToImplicitModel
from rlkit.core import logger

# 2D point mass
PATH = '/home/vitchyr/git/railrl/data/local/01-30-dev-mpc-neural-networks/01-30-dev-mpc-neural-networks_2018_01_30_11_28_28_0000--s-24549/params.pkl'
GOAL_SLICE = slice(0, 2)

# Reacher 7dof
PATH = '/home/vitchyr/git/railrl/data/local/01-27-reacher-full-mpcnn-H1/01-27-reacher-full-mpcnn-H1_2018_01_27_17_59_04_0000--s-96642/params.pkl'
GOAL_SLICE = slice(0, 7)
# GOAL_SLICE = slice(14, 17)
MULTITASK_GOAL_SLICE = GOAL_SLICE

# 2D point w/ U-wall
PATH = '/home/vitchyr/git/railrl/data/local/02-19-dev-mpc-neural-networks/02-19-dev-mpc-neural-networks_2018_02_19_22_21_24_0000--s-99698/params.pkl'
GOAL_SLICE = slice(0, 2)
MULTITASK_GOAL_SLICE = GOAL_SLICE


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str,
                        default=PATH,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--T', type=int, default=3,
                        help='Planning Horizon')
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

    # For Point2d u-shaped wall
    # import matplotlib.pyplot as plt
    # plt.show()
    # plt.ion()

    data = joblib.load(args.file)
    env = data['env']
    model = data['model']

    if args.pause:
        import ipdb; ipdb.set_trace()

    implicit_model = ModelToImplicitModel(
        model,
        # bias=-2
        order=2,  # Note: lbfgs doesn't work if the order is 1
    )
    optimizer = args.opt
    planning_horizon = args.T
    print("Optimizer choice: ", optimizer)
    if optimizer == 'slsqp':
        policy = SlsqpCMC(
            implicit_model,
            env,
            GOAL_SLICE,
            MULTITASK_GOAL_SLICE,
            solver_params={
                'ftol': 1e-3,
                'maxiter': 100,
            },
            planning_horizon=planning_horizon,
        )
    elif optimizer == 'gradient':
        policy = GradientCMC(
            implicit_model,
            env,
            GOAL_SLICE,
            MULTITASK_GOAL_SLICE,
            planning_horizon=planning_horizon,
            # For reacher, 0.1, 1, and 10 all work
            lagrange_multiplier=0.1,
            num_grad_steps=100,
            num_particles=128,
            warm_start=False,  # doesn't seem to help. maybe hurts
        )
    elif optimizer == 'state':
        policy = StateGCMC(
            implicit_model,
            env,
            GOAL_SLICE,
            MULTITASK_GOAL_SLICE,
            planning_horizon=planning_horizon,
            lagrange_multiplier=1000,
            num_grad_steps=100,
            num_particles=128,
            warm_start=False,
        )
    elif optimizer == 'lbfgs':
        policy = LBfgsBCMC(
            implicit_model,
            env,
            GOAL_SLICE,
            MULTITASK_GOAL_SLICE,
            lagrange_multipler=100,
            # warm_start=True,
            # lagrange_multipler=1,
            planning_horizon=planning_horizon,
            solver_params={
                'factr': 1e9,
                # 'factr': 1e12,
            },
        )

    paths = []
    while True:
        paths.append(rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        ))
        env.log_diagnostics(paths)
        logger.dump_tabular()
