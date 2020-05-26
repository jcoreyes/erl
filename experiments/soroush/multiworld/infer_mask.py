from multiworld.envs.pygame.pick_and_place import (
    PickAndPlaceEnv,
    PickAndPlace1DEnv,
)

import time
import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def print_matrix(matrix, format="signed", threshold=0.1, normalize=False):
    if normalize:
        matrix = matrix.copy() / np.max(np.abs(matrix))

    assert format in ["signed", "raw"]

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if format == "raw":
                value = matrix[i][j]
            elif format == "signed":
                if np.abs(matrix[i][j]) > threshold:
                    value = 1 * np.sign(matrix[i][j])
                else:
                    value = 0
            if format == "signed":
                print(int(value), end=", ")
            else:
                if value > 0:
                    print("", end=" ")
                print("{:.2f}".format(value), end=" ")
        print()
    print()

def plot_Gaussian(mu, Sigma, dims=[0, 1], pt1=None, pt2=None):
    x, y = np.mgrid[-5:5:.05, -5:5:.05]
    pos = np.dstack((x, y))
    mu = mu[dims]
    Sigma = Sigma[dims][:,dims]
    rv = multivariate_normal(mu, Sigma)
    fig2 = plt.figure(figsize=(6, 6))
    ax2 = fig2.add_subplot(111)
    ax2.contourf(x, y, rv.logpdf(pos))

    if pt1 is not None:
        pt1 = pt1[dims]
        plt.scatter([pt1[0]], [pt1[1]])

    if pt2 is not None:
        pt2 = pt2[dims]
        plt.scatter([pt2[0]], [pt2[1]])

    plt.show()

env_class=PickAndPlaceEnv
env_kwargs=dict(
    # Environment dynamics
    action_scale=1.0,
    ball_radius=0.75, #1.
    boundary_dist=4,
    object_radius=0.50,
    min_grab_distance=0.5,
    walls=None,
    # Rewards
    action_l2norm_penalty=0,
    reward_type="dense", #dense_l1
    success_threshold=0.60,
    # Reset settings
    fixed_goal=None,
    # Visualization settings
    images_are_rgb=True,
    render_dt_msec=0,
    render_onscreen=False,
    render_size=84,
    show_goal=True,
    # get_image_base_render_size=(48, 48),
    # Goal sampling
    goal_samplers=None,
    goal_sampling_mode='random',
    num_presampled_goals=10000,
    object_reward_only=True,

    init_position_strategy='random',

    num_objects=4,
)

env = env_class(**env_kwargs)

def get_cond_distr(mu, sigma, y):
    x_dim = mu.size - y.size

    mu_x = mu[:x_dim]
    mu_y = mu[x_dim:]

    sigma_xx = sigma[:x_dim, :x_dim]
    sigma_yy = sigma[x_dim:, x_dim:]
    sigma_xy = sigma[:x_dim, x_dim:]
    sigma_yx = sigma[x_dim:, :x_dim]

    # print(sigma_xx)
    # print(sigma_yy)
    # print(sigma_xy)
    # print(sigma_yx)
    # print()

    # w, v = np.linalg.eig(sigma_yy)
    sigma_yy_inv = linalg.inv(sigma_yy)

    mu_xgy = mu_x + sigma_xy @ sigma_yy_inv @ (y - mu_y)
    sigma_xgy = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_yx
    return mu_xgy, sigma_xgy

n = 1000
show_dataset = False
states = []
goals = []
list_of_waypoints = [[], [], []]
obs_noise = 0.1
for i in range(n):
    obs_dict = env.reset()
    obs = obs_dict['state_observation']
    goal = obs_dict['state_desired_goal']

    wp1 = obs.copy()
    wp1[0:2] = obs[2:4]

    wp2 = obs.copy()
    wp2[0:2] = goal[2:4]
    wp2[2:4] = goal[2:4]
    # wp2[4:6] = goal[4:6]

    wp3 = obs.copy()
    wp3[0:2] = goal[0:2]
    wp3[2:4] = goal[2:4]

    wp1 += np.random.normal(0, obs_noise, obs.size)
    wp2 += np.random.normal(0, obs_noise, obs.size)
    wp3 += np.random.normal(0, obs_noise, obs.size)
    obs += np.random.normal(0, obs_noise, obs.size)
    goal += np.random.normal(0, obs_noise, obs.size)

    list_of_waypoints[0].append(wp1)
    list_of_waypoints[1].append(wp2)
    list_of_waypoints[2].append(wp3)
    goals.append(goal)
    states.append(obs)

    if show_dataset:
        env.set_to_goal({
            'state_desired_goal': wp1
        })
        env.render()
        time.sleep(2)

list_of_waypoints = np.array(list_of_waypoints)
goals = np.array(goals)
states = np.array(states)

for waypoints in list_of_waypoints[:1]:
    for i in range(1):
        state = states[i]
        goal = goals[i]
        waypoint = waypoints[i]

        # mu = np.mean(np.concatenate((waypoints, states), axis=1), axis=0)
        # sigma = np.cov(np.concatenate((waypoints, states), axis=1).T)
        # mu_w_given_g, sigma_w_given_g = get_cond_distr(mu, sigma, state)

        mu = np.mean(np.concatenate((waypoints, goals), axis=1), axis=0)
        sigma = np.cov(np.concatenate((waypoints, goals), axis=1).T)
        mu_w_given_g, sigma_w_given_g = get_cond_distr(mu, sigma, goal)

        # mu = np.mean(np.concatenate((waypoints, states, goals), axis=1), axis=0)
        # sigma = np.cov(np.concatenate((waypoints, states, goals), axis=1).T)
        # mu_w_given_g, sigma_w_given_g = get_cond_distr(mu, sigma, np.concatenate((state, goal)))

        # mu_w_given_g = np.mean(waypoints, axis=0)
        # sigma_w_given_g = np.cov(waypoints.T)

        w, v = np.linalg.eig(sigma_w_given_g)
        if i == 0:
            print("eig:", sorted(w))
        if np.min(sorted(w)) < 1e-6:
            eps = 1e-6
        else:
            eps = 0
        sigma_inv = linalg.inv(sigma_w_given_g + eps*np.identity(sigma_w_given_g.shape[0]))
        # sigma_inv = sigma_inv / np.max(np.abs(sigma_inv))
        print("s:", state)
        print("g:", goal)
        print("w:", mu_w_given_g)

        # print(sigma_w_given_g)
        if i == 0:
            print_matrix(
                sigma_w_given_g,
                format="raw",
                normalize=True,
                threshold=0.4,
            )
            print_matrix(
                sigma_inv,
                # format="raw",
                normalize=True,
                threshold=0.4,
            )

        # plot_Gaussian(mu_w_given_g, sigma_w_given_g, pt1=goal, pt2=state, dims=[0, 2])