import argparse
import numpy as np
import matplotlib.pyplot as plt

import joblib

from rlkit.misc.eval_util import get_generic_path_information
from rlkit.state_distance.rollout_util import multitask_rollout
from rlkit.core import logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='path to the snapshot file')
    parser.add_argument('--nrolls', type=int, default=1,
                        help='Number of rollout per eval')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--ndc', help='not (decrement and cycle tau)',
                        action='store_true')
    args = parser.parse_args()

    data = joblib.load(args.file)
    env = data['env']
    if 'policy' in data:
        policy = data['policy']
    else:
        policy = data['exploration_policy']
    policy.train(False)
    plot_performance(policy, env, args.nrolls)
    plot_performance(policy, env, args.nrolls)


def plot_performance(policy, env, nrolls):
    print("max_tau, distance")
    # fixed_goals = [-40, -30, 30, 40]
    fixed_goals = [-5, -3, 3, 5]
    taus = np.arange(10) * 10
    for row, fix_tau in enumerate([True, False]):
        for col, horizon_fixed in enumerate([True, False]):
            plot_num = row + 2*col + 1
            plt.subplot(2, 2, plot_num)
            for fixed_goal in fixed_goals:
                distances = []
                for max_tau in taus:
                    paths = []
                    for _ in range(nrolls):
                        goal = env.sample_goal_for_rollout()
                        goal[0] = fixed_goal
                        path = multitask_rollout(
                            env,
                            policy,
                            goal,
                            init_tau=max_tau,
                            max_path_length=100 if horizon_fixed else max_tau + 1,
                            animated=False,
                            cycle_tau=True,
                            decrement_tau=not fix_tau,
                        )
                        paths.append(path)
                    env.log_diagnostics(paths)
                    for key, value in get_generic_path_information(paths).items():
                        logger.record_tabular(key, value)
                    distance = float(dict(logger._tabular)['Final Distance to goal Mean'])
                    distances.append(distance)

                plt.plot(taus, distances)
                print("line done")
            plt.legend([str(goal) for goal in fixed_goals])
            if fix_tau:
                plt.xlabel("Tau (Horizon-1)")
            else:
                plt.xlabel("Initial tau (=Horizon-1)")
            plt.xlabel("Max tau")
            plt.ylabel("Final distance to goal")
            plt.title("Fix Tau = {}, Horizon Fixed to 100  = {}".format(
                fix_tau,
                horizon_fixed,
            ))
    plt.show()
    plt.savefig('results/iclr2018/cheetah-sweep-tau-eval-5-3.jpg')
if __name__ == "__main__":
    main()
