import argparse
import joblib
import torch

from rlkit.core import logger
from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule

PATH = '/home/vitchyr/git/railrl/data/local/01-27-reacher-full-mpcnn-H1/01-27-reacher-full-mpcnn-H1_2018_01_27_17_59_04_0000--s-96642/params.pkl'


class ModelToTdm(PyTorchModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, goal, taus, action):
        out = (goal - obs - self.model(obs, action)) ** 2
        return -torch.norm(out, dim=1).unsqueeze(1)


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
                final_distance = logger.get_table_dict()[
                    'Final Euclidean distance to goal Mean']
                print("final distance", final_distance)
                # logger.dump_tabular()
