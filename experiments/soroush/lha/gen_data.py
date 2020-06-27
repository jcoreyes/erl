from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC

import time
from tqdm import tqdm
import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def get_env(num_obj=4, render=False):
    env_class=SawyerLiftEnvGC
    env_kwargs={
        'action_scale': .06, #.06
        'action_repeat': 10, #5
        'timestep': 1./120, #1./240 1/120
        'max_force': 1000, #1000
        'solver_iterations': 500, #150

        'gui': True,  # False,
        'goal_mult': 0,
        'pos_init': [.75, -.3, 0],
        'pos_high': [.75, .4, .3], #[.75, .4, .3],
        'pos_low': [.75, -.4, -.36], #[.75, -.4, -.36],
        'reset_obj_in_hand_rate': 0.0, #0.0
        'img_dim': 48,
        'goal_mode': 'obj_in_bowl',
        'num_obj': 4, #2

        'lite_reset': True,

        # 'use_rotated_gripper': True, #False
        # 'use_wide_gripper': False, #False
        # 'soft_clip': True,
        # 'obj_urdf': 'spam_long',
        # 'max_joint_velocity': 2.0,

        # 'use_rotated_gripper': False,  # False
        # 'use_wide_gripper': False,  # False
        # 'soft_clip': False,
        # 'obj_urdf': 'spam',
        # 'max_joint_velocity': None,

        'use_rotated_gripper': True,  # False
        'use_wide_gripper': True,  # False
        'soft_clip': True,
        'obj_urdf': 'spam',
        'max_joint_velocity': None,
    }
    env_kwargs['gui'] = render
    env_kwargs['num_obj'] = num_obj
    env = env_class(**env_kwargs)
    return env

def set_wp(wp, obs, goal, mode='hand_to_obj', obj_id=0):
    obj_start_idx = 2 * obj_id + 2
    obj_end_idx = obj_start_idx + 2
    if mode == 'hand_to_obj':
        wp[0:2] = obs[obj_start_idx:obj_end_idx]
    elif mode == 'obj_to_goal':
        wp[obj_start_idx:obj_end_idx] = goal[obj_start_idx:obj_end_idx]
    elif mode == 'obj_and_hand_to_air':
        wp[obj_start_idx+1] = -0.20
        wp[0:2] = wp[obj_start_idx:obj_end_idx]
    elif mode == 'obj_and_hand_to_goal':
        wp[obj_start_idx:obj_end_idx] = goal[obj_start_idx:obj_end_idx]
        wp[0:2] = wp[obj_start_idx:obj_end_idx]
    else:
        raise NotImplementedError

def gen_dataset(
        num_obj=4,
        n=50,
        render=False,
        hand_to_obj=False,
        obj_and_hand_to_air=True,
        obj_to_goal=True,
        obj_and_hand_to_goal=False,
        cumulative=False,
        randomize_objs=False,
):
    assert not (obj_to_goal and obj_and_hand_to_goal)

    env = get_env(num_obj=num_obj, render=render)
    states = []
    goals = []

    stages = []
    if hand_to_obj:
        stages.append('hand_to_obj')
    if obj_and_hand_to_air:
        stages.append('obj_and_hand_to_air')
    if obj_to_goal:
        stages.append('obj_to_goal')
    if obj_and_hand_to_goal:
        stages.append('obj_and_hand_to_goal')
    num_stages_per_obj = len(stages)
    num_wps = num_stages_per_obj * num_obj

    list_of_waypoints = []
    t1 = time.time()
    print("Generating dataset...")
    for i in tqdm(range(n)):
        list_of_waypoints.append([])
        obj_ids = [i for i in range(num_obj)]
        if randomize_objs:
            np.random.shuffle(obj_ids)

        obs_dict = env.reset()
        obs = obs_dict['state_achieved_goal'] #'state_observation'
        goal = obs_dict['state_desired_goal']

        goals.append(goal)
        states.append(obs)

        if render:
            env.render()
            time.sleep(2)

        if cumulative:
            wp = obs.copy()
        for j in range(num_wps):
            if not cumulative:
                wp = obs.copy()

            obj_idx = j // num_stages_per_obj
            stage_idx = j % num_stages_per_obj

            set_wp(wp, obs, goal, mode=stages[stage_idx], obj_id=obj_ids[obj_idx])

            list_of_waypoints[i].append(wp)

            if render:
                wp = list_of_waypoints[i][j]
                env.set_to_goal({
                    'state_desired_goal': wp
                })
                env.render()
                time.sleep(2)

        if render:
            env.render()
            time.sleep(2)

    list_of_waypoints = np.array(list_of_waypoints)
    goals = np.array(goals)
    states = np.array(states)

    print("Done. Time:", time.time() - t1)

    return list_of_waypoints, goals, states

def get_cond_distr(mu, sigma, y):
    x_dim = mu.size - y.size

    mu_x = mu[:x_dim]
    mu_y = mu[x_dim:]

    sigma_xx = sigma[:x_dim, :x_dim]
    sigma_yy = sigma[x_dim:, x_dim:]
    sigma_xy = sigma[:x_dim, x_dim:]
    sigma_yx = sigma[x_dim:, :x_dim]

    sigma_yy_inv = linalg.inv(sigma_yy)

    mu_xgy = mu_x + sigma_xy @ sigma_yy_inv @ (y - mu_y)
    sigma_xgy = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_yx
    return mu_xgy, sigma_xgy

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

def plot_Gaussian(mu, Sigma, list_of_dims=[[0, 1], [2, 3], [0, 2], [1, 3]], pt1=None, pt2=None):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    x, y = np.mgrid[-0.5:0.5:.01, -0.5:0.5:.01]
    pos = np.dstack((x, y))
    for i in range(len(list_of_dims)):
        dims = list_of_dims[i]
        rv = multivariate_normal(mu[dims], Sigma[dims][:,dims])

        plt_idx1 = i // 2
        plt_idx2 = i % 2
        axs[plt_idx1, plt_idx2].contourf(x, y, rv.logpdf(pos))
        axs[plt_idx1, plt_idx2].set_title(str(dims))

        if pt1 is not None:
            axs[plt_idx1, plt_idx2].scatter([pt1[dims][0]], [pt1[dims][1]])

        if pt2 is not None:
            axs[plt_idx1, plt_idx2].scatter([pt2[dims][0]], [pt2[dims][1]])

    plt.show()

num_sets = 50
list_of_waypoints, goals, states = gen_dataset(
    num_obj=4,
    n=num_sets,
    render=False,
    hand_to_obj=False,
    obj_and_hand_to_air=False,
    obj_to_goal=True,
    obj_and_hand_to_goal=False,
    cumulative=False,
    randomize_objs=False,
)

### Add noise to waypoints ###
obs_noise=0.01
list_of_waypoints += np.random.normal(0, obs_noise, list_of_waypoints.shape)
goals += np.random.normal(0, obs_noise, goals.shape)
states += np.random.normal(0, obs_noise, states.shape)

num_subtasks = list_of_waypoints.shape[1]
for i in range(num_subtasks):
    waypoints = list_of_waypoints[:,i,:]

    mu = np.mean(np.concatenate((waypoints, goals), axis=1), axis=0)
    sigma = np.cov(np.concatenate((waypoints, goals), axis=1).T)

    for j in range(1):
        state = states[j]
        goal = goals[j]
        mu_w_given_g, sigma_w_given_g = get_cond_distr(mu, sigma, goal)
        plot_Gaussian(
            mu_w_given_g, sigma_w_given_g,
            pt1=goal, pt2=state,
            # list_of_dims=[[0, 1], [2, 3], [0, 2], [1, 3]],
            list_of_dims=[[2, 3], [4, 5], [2, 4], [3, 5]],
        )

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
        if j == 0:
            # print_matrix(
            #     sigma_w_given_g,
            #     format="raw",
            #     normalize=True,
            #     threshold=0.4,
            # )
            print_matrix(
                sigma_inv,
                format="raw",
                normalize=True,
                threshold=0.4,
            )