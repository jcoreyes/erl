import argparse
import joblib

from railrl.samplers.util import rollout
from railrl.torch.mpc.collocation.collocation_mpc_controller import SlsqpCMC, \
    GradientCMC, StateGCMC, LBfgsBCMC
from railrl.torch.mpc.collocation.model_to_implicit_model import \
    ModelToImplicitModel
from railrl.core import logger

# 2D point mass
PATH = '/home/vitchyr/git/railrl/data/local/01-30-dev-mpc-neural-networks/01-30-dev-mpc-neural-networks_2018_01_30_11_28_28_0000--s-24549/params.pkl'
GOAL_SLICE = slice(0, 2)
# Reacher 7dof
PATH = '/home/vitchyr/git/railrl/data/local/01-27-reacher-full-mpcnn-H1/01-27-reacher-full-mpcnn-H1_2018_01_27_17_59_04_0000--s-96642/params.pkl'
GOAL_SLICE = slice(0, 7)


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
    model = data['model']

    implicit_model = ModelToImplicitModel(
        model,
        # bias=-2
        order=2,  # Note: lbfgs doesn't work if the order is 1
    )
    optimizer = args.opt
    print("Optimizer choice: ", optimizer)
    if optimizer == 'slsqp':
        policy = SlsqpCMC(
            implicit_model,
            env,
            GOAL_SLICE,
            solver_params={
                'ftol': 1e-3,
                'maxiter': 100,
            },
            planning_horizon=1,
        )
    elif optimizer == 'gradient':
        policy = GradientCMC(
            implicit_model,
            env,
            GOAL_SLICE,
            planning_horizon=1,
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
            planning_horizon=1,
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
            lagrange_multipler=10,
            planning_horizon=1,
            solver_params={
                'factr': 1e9,
            },
        )

    while True:
        paths = [rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        )]
        env.log_diagnostics(paths)
        logger.dump_tabular()
