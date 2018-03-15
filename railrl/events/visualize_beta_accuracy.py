import argparse
import numpy as np
import matplotlib.pyplot as plt

import joblib

from railrl.misc.visualization_util import make_heat_map, plot_heatmap
from railrl.policies.simple import RandomPolicy
from railrl.state_distance.rollout_util import multitask_rollout

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()
    if args.pause:
        import ipdb; ipdb.set_trace()

    file = args.file
    data = joblib.load(file)
    beta_q = data['beta_q']
    env = data['env']

    random_policy = RandomPolicy(env.action_space)
    path = multitask_rollout(
        env,
        random_policy,
        init_tau=0,
        max_path_length=2,
        # animated=True,
    )
    path_obs = path['observations']
    path_actions = path['actions']
    path_next_obs = path['next_observations']
    num_steps_left = np.zeros((1, 1))

    obs = path_obs[0:1]
    next_obs = path_next_obs[0:1]
    actions = path_actions[0:1]
    beta_values = []

    def create_beta_eval(obs, goal):
        def beta_eval(a1, a2):
            actions = np.array([[a1, a2]])
            return beta_q.eval_np(
                observations=np.array([[
                    *obs
                ]]),
                actions=actions,
                goals=np.array([[
                    *goal
                ]]),
                num_steps_left=num_steps_left
            )[0, 0]
        return beta_eval

    def create_goal_eval(action, pos):
        def goal_eval(x1, x2):
            actions = np.array([[*action]])
            return beta_q.eval_np(
                observations=np.array([[
                    *pos
                ]]),
                actions=actions,
                goals=np.array([[
                    x1, x2
                ]]),
                num_steps_left=num_steps_left
            )[0, 0]
        return goal_eval

    print("obs:", obs)
    print("true action:", actions)
    print("next obs:", next_obs)

    obs = (4, 4)
    goal = (3, 4)
    heatmap = make_heat_map(create_beta_eval(obs, goal),
                            [-1, 1], [-1, 1], resolution=50)
    plot_heatmap(heatmap)
    plt.title("pos {}. goal {}".format(obs, goal))

    plt.figure()
    obs = (0, 0)
    goal = (1, 0)
    heatmap = make_heat_map(create_beta_eval(obs, goal),
                            [-1, 1], [-1, 1], resolution=50)
    plot_heatmap(heatmap)
    plt.title("pos {}. goal {}".format(obs, goal))

    if False:
        pos = (4, 4)
        plt.figure()
        right_action_eval = create_goal_eval((1, 0), pos)
        heatmap = make_heat_map(right_action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)
        plt.title("+1, 0")

        plt.figure()
        left_action_eval = create_goal_eval((-1, 0), pos)
        heatmap = make_heat_map(left_action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)
        plt.title("-1, 0")

        plt.figure()
        down_action_eval = create_goal_eval((0, 1), pos)
        heatmap = make_heat_map(down_action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)
        plt.title("0, +1")

        plt.figure()
        up_action_eval = create_goal_eval((0, -1), pos)
        heatmap = make_heat_map(up_action_eval, [-4, 4], [-4, 4], resolution=50)
        plot_heatmap(heatmap)
        plt.title("0, -1")

    plt.show()
