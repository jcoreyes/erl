import argparse

import joblib

import torch
import numpy as np
from railrl.core import logger
from railrl.samplers.util import rollout
from railrl.state_distance.probabilistic_tdm.mpc_controller import \
    ImplicitMPCController
from railrl.state_distance.rollout_util import multitask_rollout
from railrl.torch.core import PyTorchModule

PATH = '/home/vitchyr/git/railrl/data/doodads3/01-21-reacher-full-sac-tdm/01' \
       '-21-reacher-full-sac-tdm-id4-s9639/params.pkl'

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
    tdm = data['qf']
    num_samples = 1000
    resolution = 10
    if 'policy' in data:
        original_policy = data['policy']
    else:
        original_policy = data['exploration_policy'].policy
    original_policy.env = env
    original_policy.cost_fn = env.cost_fn
    original_policy.num_simulated_paths = args.npath
    original_policy.horizon = 1
    if args.justsim:
        while True:
            goal = np.array(
                [-0.20871628403863521, -0.0026045399886658986,
                 1.5508042141054157, -1.4642474683183448, 0.078682316483737469,
                 -0.49380223494132874, -1.4292323965597007,
                 0.098066894378607036, -0.26046187123103803, 1.526653353350421,
                 3.0780086804131308, -0.53339687898388422, -2.579676257728218,
                 -4.9314019794438844, 0.38974402757384086, -1.1045324518922441,
                 0.010756958159845592]
            )
            # goal = np.array([
            #     -0.29421230153709033, 0.038686863527214843, 1.6602570424019201,
            #      0.0059356156399937325, -0.0064939457331620459,
            #      -0.9692505876434705, -1.5013519244203266, 0.26682933070687942,
            #      -0.083162869319415134, -1.3329693169147059,
            #      -0.1843069709628351, 1.0109360204751949, -0.20689527928910664,
            #      0.020834381975244821, 0.81598804213626219,
            #      -0.93234483757944919, -0.037532679060846452
            # ])
            env.set_goal(goal)
            # path = rollout(
            #     env,
            #     original_policy,
            #     max_path_length=args.H,
            #     animated=not args.hide,
            # )
            path = multitask_rollout(
                env,
                original_policy,
                # env.multitask_goal,
                goal,
                tau=0,
                max_path_length=args.H,
                animated=not args.hide,
                cycle_tau=False,
                decrement_tau=False,
            )
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics([path])
            logger.dump_tabular()
    else:
        for weight in [0.01]:
            for num_simulated_paths in [args.npath]:
                print("")
                print("weight", weight)
                print("num_simulated_paths", num_simulated_paths)
                policy = ImplicitMPCController(
                    env,
                    tdm,
                    original_policy,
                    num_simulated_paths=num_simulated_paths,
                    feasibility_weight=weight,
                )
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
