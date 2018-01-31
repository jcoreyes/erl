import argparse
import joblib

from railrl.samplers.util import rollout
from railrl.torch.mpc.collocation.collocation_mpc_controller import SlsqpCMC
from railrl.torch.mpc.collocation.model_to_implicit_model import \
    ModelToImplicitModel

PATH = '/home/vitchyr/git/railrl/data/local/01-30-dev-mpc-neural-networks/01-30-dev-mpc-neural-networks_2018_01_30_11_28_28_0000--s-24549/params.pkl'

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
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    data = joblib.load(args.file)
    env = data['env']
    model = data['model']

    implicit_model = ModelToImplicitModel(model, bias=-2)
    solver_params = {
        'ftol': 1e-2,
    }
    policy = SlsqpCMC(
        implicit_model,
        env,
        use_implicit_model_gradient=True,
        solver_params=solver_params
    )
    while True:
        paths = [rollout(
            env,
            policy,
            max_path_length=args.H,
            animated=not args.hide,
        )]
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics(paths)
