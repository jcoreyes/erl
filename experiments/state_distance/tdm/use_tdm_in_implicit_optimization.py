import argparse

import joblib

from rlkit.core import logger
from rlkit.samplers.util import rollout
from rlkit.torch.mpc.collocation import \
    CollocationMpcController
from rlkit.state_distance.rollout_util import multitask_rollout
from rlkit.state_distance.util import merge_into_flat_obs
from rlkit.torch.core import PyTorchModule

PATH = '/home/vitchyr/git/railrl/data/local/01-22-dev-sac-tdm-launch/01-22-dev-sac-tdm-launch_2018_01_22_13_31_47_0000--s-3096/params.pkl'
PATH = '/home/vitchyr/git/railrl/data/doodads3/01-23-reacher-full-ddpg' \
       '-tdm-mtau-0/01-23-reacher-full-ddpg-tdm-mtau-0-id1-s49343/params.pkl'


class ImplicitModel(PyTorchModule):
    def __init__(self, qf, vf):
        super().__init__()
        self.qf = qf
        self.vf = vf

    def forward(self, obs, goals, taus, actions):
        flat_obs = merge_into_flat_obs(obs, goals, taus)
        if self.vf is None:
            return self.qf(flat_obs, actions)
        else:
            return self.qf(flat_obs, actions) - self.vf(flat_obs)


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
    qf = data['qf']
    if 'vf' in data:
        vf = data['vf']
    else:
        vf = None
    tdm = ImplicitModel(qf, vf)
    implicit_model = ImplicitModel(qf, vf)
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
            # goal = np.array(
            #     [-0.20871628403863521, -0.0026045399886658986,
            #      1.5508042141054157, -1.4642474683183448, 0.078682316483737469,
            #      -0.49380223494132874, -1.4292323965597007,
            #      0.098066894378607036, -0.26046187123103803, 1.526653353350421,
            #      3.0780086804131308, -0.53339687898388422, -2.579676257728218,
            #      -4.9314019794438844, 0.38974402757384086, -1.1045324518922441,
            #      0.010756958159845592]
            # )
            # goal = np.array([ 0.00934609, -0.06385207,  1.1130754 , -1.791122  ,  1.07486696, -0.44234793, -1.30457667, -0.28577358,  0.45736275, -2.42824523, -2.32354267, -2.23998136, -0.82785123, -0.53785655,  0.39529478, -0.97393436,  0.15426666])
            # goal = np.array([
            #     -0.29421230153709033, 0.038686863527214843, 1.6602570424019201,
            #      0.0059356156399937325, -0.0064939457331620459,
            #      -0.9692505876434705, -1.5013519244203266, 0.26682933070687942,
            #      -0.083162869319415134, -1.3329693169147059,
            #      -0.1843069709628351, 1.0109360204751949, -0.20689527928910664,
            #      0.020834381975244821, 0.81598804213626219,
            #      -0.93234483757944919, -0.037532679060846452
            # ])
            # env.set_goal(goal)
            # path = rollout(
            #     env,
            #     original_policy,
            #     max_path_length=args.H,
            #     animated=not args.hide,
            # )
            # goal = np.array([1.4952445864440109, 0.058365245652776926,
            #                  1.3854542196239863, -0.64643021271356582, 0.25729402753586905, -1.0559116816553138, -1.2942449012062724, 0.84327192781565719, -0.18665817808605106, 0.28887389778176836, -4.1567137920511996, -0.25677653709657877, 1.2789295463658288, 0.47291580030348057, 0.34130661157042974, 0.13003414588968379, -0.009319281912785882])
            # goal = np.array([1.6958471372903317, 0.2122816058111654,
            #                  0.29760944600589051, -0.016908392188567031, -0.58501650189613841, -0.018928029822669078, -1.2091424324357098, 0.16575094693524303, 0.32991058173255483, 2.8226738796936663, -0.57674228567507868, 1.5591211986667852, 0.53321884401877584, -3.8082528691546091, -0.11086735631355096, 0.29765427337121497, -0.16364599916575717])
            goal = env.sample_goal_for_rollout()
            goal[7:14] = 0
            path = multitask_rollout(
                env,
                original_policy,
                # env.multitask_goal,
                goal,
                init_tau=10,
                max_path_length=args.H,
                animated=not args.hide,
                cycle_tau=True,
                decrement_tau=False,
            )
            if hasattr(env, "log_diagnostics"):
                env.log_diagnostics([path])
            logger.dump_tabular()
    else:
        for weight in [1]:
            for num_simulated_paths in [args.npath]:
                print("")
                print("weight", weight)
                print("num_simulated_paths", num_simulated_paths)
                policy = CollocationMpcController(
                    env,
                    implicit_model,
                    original_policy,
                    num_simulated_paths=num_simulated_paths,
                    feasibility_weight=weight,
                )
                policy.train(False)
                paths = []
                for _ in range(5):
                    goal = env.sample_goal_for_rollout()
                    env.set_goal(goal)
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
