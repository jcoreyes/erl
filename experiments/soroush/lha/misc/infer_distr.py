from roboverse.envs.goal_conditioned.sawyer_lift_gc import SawyerLiftEnvGC

import time
from tqdm import tqdm
import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def get_env(num_obj=4, render=False, lite_reset=False):
    env_class=SawyerLiftEnvGC
    env_kwargs={
        'action_scale': .06,
        'action_repeat': 10,
        'timestep': 1./120,
        'solver_iterations': 500,
        'max_force': 1000,

        'gui': True,
        'pos_init': [.75, -.3, 0],
        'pos_high': [.75, .4, .3],
        'pos_low': [.75, -.4, -.36],
        'reset_obj_in_hand_rate': 0.0,
        'goal_sampling_mode': 'ground',
        'random_init_bowl_pos': True,
        'bowl_type': 'fixed',
        'bowl_bounds': [-0.40, 0.40],

        'hand_reward': True,
        'gripper_reward': True,
        'bowl_reward': True,

        'use_rotated_gripper': True,
        'use_wide_gripper': True,
        'soft_clip': True,
        'obj_urdf': 'spam',
        'max_joint_velocity': None,
    }
    env_kwargs['gui'] = render
    env_kwargs['num_obj'] = num_obj
    env_kwargs['lite_reset'] = lite_reset
    env = env_class(**env_kwargs)
    return env

def set_wp(wp, obs, goal, env, mode='hand_to_obj', obj_id=0, other_dims_random=False):
    obj_start_idx = 2 * obj_id + 2
    obj_end_idx = obj_start_idx + 2
    dims = np.arange(obj_start_idx, obj_end_idx)
    if mode == 'hand_to_obj':
        wp[0:2] = wp[obj_start_idx:obj_end_idx]
        dims = np.concatenate((dims, np.arange(0, 2)))
    elif mode == 'obj_to_goal':
        wp[obj_start_idx:obj_end_idx] = goal[obj_start_idx:obj_end_idx]
    elif mode == 'obj_and_hand_to_air':
        wp[obj_start_idx+1] = -0.20
        wp[0:2] = wp[obj_start_idx:obj_end_idx]
        dims = np.concatenate((dims, np.arange(0, 2)))
    elif mode == 'obj_and_hand_to_goal':
        wp[obj_start_idx:obj_end_idx] = goal[obj_start_idx:obj_end_idx]
        wp[0:2] = wp[obj_start_idx:obj_end_idx]
        dims = np.concatenate((dims, np.arange(0, 2)))
    elif mode == 'obj_to_bowl':
        bowl_dim = len(wp) - 2
        wp[obj_start_idx] = wp[bowl_dim]
        dims = np.concatenate((dims, [bowl_dim]))
    else:
        raise NotImplementedError

    other_dims = [d for d in np.arange(len(wp)) if d not in dims]
    if other_dims_random:
        wp[other_dims] = env.observation_space.spaces['state_achieved_goal'].sample()[other_dims]

def gen_dataset(
        num_obj=4,
        obj_ids=None,
        n=50,
        render=False,
        lite_reset=False,
        hand_to_obj=False,
        obj_and_hand_to_air=False,
        obj_to_goal=False,
        obj_and_hand_to_goal=False,
        obj_to_bowl=False,
        cumulative=False,
        randomize_objs=False,
        other_dims_random=True,
):
    assert not (obj_to_goal and obj_and_hand_to_goal)

    env = get_env(num_obj=num_obj, render=render, lite_reset=lite_reset)
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
    if obj_to_bowl:
        stages.append('obj_to_bowl')
    num_stages_per_obj = len(stages)

    if obj_ids is None:
        obj_ids = [i for i in range(num_obj)]
    num_wps = num_stages_per_obj * len(obj_ids)

    list_of_waypoints = []
    t1 = time.time()
    print("Generating dataset...")
    for i in tqdm(range(n)):
        list_of_waypoints.append([])
        if randomize_objs:
            np.random.shuffle(obj_ids)

        obs_dict = env.reset()
        obs = obs_dict['state_achieved_goal'] #'state_observation'
        goal = obs_dict['state_desired_goal']

        goals.append(goal)
        states.append(obs)

        if render:
            env.set_to_goal({
                'state_desired_goal': obs
            })
            env.render()
            time.sleep(5)

            env.set_to_goal({
                'state_desired_goal': goal
            })
            env.render()
            time.sleep(5)

        if cumulative:
            wp = obs.copy()
        for j in range(num_wps):
            if not cumulative:
                wp = obs.copy()

            obj_idx = j // num_stages_per_obj
            stage_idx = j % num_stages_per_obj

            set_wp(
                wp, obs, goal, env,
                mode=stages[stage_idx],
                obj_id=obj_ids[obj_idx],
                other_dims_random=other_dims_random,
            )

            list_of_waypoints[i].append(wp)

            if render:
                wp = list_of_waypoints[i][j]
                env.set_to_goal({
                    'state_desired_goal': wp
                })
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

def print_matrix(matrix, format="signed", threshold=0.1, normalize=False, precision=5):
    if normalize:
        matrix = matrix.copy() / np.max(np.abs(matrix))

    assert format in ["signed", "raw"]
    assert precision in [2, 5, 10]

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
                if precision == 2:
                    print("{:.2f}".format(value), end=" ")
                elif precision == 5:
                    print("{:.5f}".format(value), end=" ")
                elif precision == 10:
                    print("{:.10f}".format(value), end=" ")
        print()
    print()

# env settings
num_sets = 30 #500
num_obj = 4
obj_ids = [0]

# data generation settings
use_cached_data = True
lite_reset = False
render = False

# inference settings
obs_noise = 0.01 #0.01
correlated_noise = False
cond_num = 1e2
context_conditioned = False
vis_distr = False

if not use_cached_data:
    ### generate and save the data
    list_of_waypoints, goals, states = gen_dataset(
        num_obj=num_obj,
        obj_ids=obj_ids,
        n=num_sets,
        render=render,
        lite_reset=lite_reset,
        # hand_to_obj=True,
        # obj_and_hand_to_air=True,
        # obj_to_goal=True,
        # obj_and_hand_to_goal=True,
        obj_to_bowl=True,
        cumulative=False,
        randomize_objs=False,
        other_dims_random=True,
    )
    np.save(
        'num_obj={}.npy'.format(num_obj),
        {
            'list_of_waypoints': list_of_waypoints,
            'goals': goals,
            'states': states,
        }
    )
else:
    ### load the data
    data = np.load('num_obj={}_cached.npy'.format(num_obj))[()]
    list_of_waypoints = data['list_of_waypoints']
    goals = data['goals']
    states = data['states']
    indices = np.arange(len(list_of_waypoints))
    np.random.shuffle(indices)
    assert len(indices) >= num_sets
    indices = indices[:num_sets]
    list_of_waypoints = list_of_waypoints[indices]
    goals = goals[indices]
    states = states[indices]

num_subtasks = list_of_waypoints.shape[1]
print("num subtasks:", num_subtasks)

### Add noise to waypoints ###
if correlated_noise:
    noise = np.random.normal(0, obs_noise, goals.shape)
    for i in range(num_subtasks):
        list_of_waypoints[:,i] += noise
    goals += noise
    states += np.random.normal(0, obs_noise, states.shape)
else:
    list_of_waypoints += np.random.normal(0, obs_noise, list_of_waypoints.shape)
    goals += np.random.normal(0, obs_noise, goals.shape)
    states += np.random.normal(0, obs_noise, states.shape)

# for i in [0]: #range(num_subtasks)
for i in range(num_subtasks):
    waypoints = list_of_waypoints[:,i,:]

    mu = np.mean(np.concatenate((waypoints, goals), axis=1), axis=0)
    sigma = np.cov(np.concatenate((waypoints, goals), axis=1).T)

    for j in range(1):
        state = states[j]
        goal = goals[j]
        goal = goal.copy()
        # goal[4:6] = goal[0:2]

        if context_conditioned:
            mu_w_given_c, sigma_w_given_c = get_cond_distr(mu, sigma, goal)
        else:
            mu_w_given_c, sigma_w_given_c = mu[:len(goal)], sigma[:len(goal),:len(goal)]

        w, v = np.linalg.eig(sigma_w_given_c)
        if j == 0:
            print("eig:", sorted(w))
            print("cond number:", np.max(w) / np.min(w))
        l, h = np.min(w), np.max(w)
        target = 1 / cond_num
        # if l < target:
        #     eps = target
        # else:
        #     eps = 0
        if (l / h) < target:
            eps = (h * target - l) / (1 - target)
        else:
            eps = 0
        sigma_w_given_c += eps * np.identity(sigma_w_given_c.shape[0])

        sigma_inv = linalg.inv(sigma_w_given_c)
        # sigma_inv = sigma_inv / np.max(np.abs(sigma_inv))
        # print(sigma_w_given_c)
        if j == 0:
            print_matrix(
                sigma_w_given_c,
                format="raw",
                normalize=True,
                threshold=0.4,
                precision=10,
            )
            print_matrix(
                # sigma_inv[[2, 3, 6]][:,[2, 3, 6]],
                sigma_inv,
                format="raw",
                normalize=True,
                threshold=0.4,
                precision=2,
            )
            # exit()

        # print("s:", state)
        # print("g:", goal)
        # print("w:", mu_w_given_c)
        # print()

        if vis_distr:
            if i == 0:
                list_of_dims = [[0, 1], [2, 3], [0, 2], [1, 3]]
            elif i == 2:
                list_of_dims = [[2, len(goal) - 2]]
            else:
                list_of_dims = [[2, 3], [4, 5], [2, 4], [3, 5]]
            plot_Gaussian(
                mu_w_given_c,
                Sigma=sigma_w_given_c,
                pt1=goal, #pt2=state,
                list_of_dims=list_of_dims,
            )