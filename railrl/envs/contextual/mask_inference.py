import numpy as np
from scipy import linalg
import time

def get_cond_distr_params(mu, sigma, x_dim, max_cond_num):
    mu_x = mu[:x_dim]
    mu_y = mu[x_dim:]

    sigma_xx = sigma[:x_dim, :x_dim]
    sigma_yy = sigma[x_dim:, x_dim:]
    sigma_xy = sigma[:x_dim, x_dim:]
    sigma_yx = sigma[x_dim:, :x_dim]

    sigma_yy_inv = linalg.inv(sigma_yy)

    mu_mat = sigma_xy @ sigma_yy_inv
    sigma_xgy = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_yx

    w, v = np.linalg.eig(sigma_xgy)
    l, h = np.min(w), np.max(w)
    target = 1 / max_cond_num
    if (l / h) < target:
        eps = (h * target - l) / (1 - target)
    else:
        eps = 0
    sigma_xgy_inv = linalg.inv(sigma_xgy + eps * np.identity(sigma_xgy.shape[0]))

    return mu_x, mu_y, mu_mat, sigma_xgy_inv

def print_matrix(matrix, format="raw", threshold=0.1, normalize=True):
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

def gen_data(env, idx_masks, n, other_dims_random=True):
    from tqdm import tqdm
    num_masks = len(idx_masks)

    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size

    goals = np.zeros((n, goal_dim))
    list_of_waypoints = np.zeros((num_masks, n, goal_dim))

    t1 = time.time()
    print("Generating dataset...")

    # data collection
    for i in tqdm(range(n)):
        obs_dict = env.reset()
        obs = obs_dict['state_achieved_goal']
        goal = obs_dict['state_desired_goal']

        for (mask_id, idx_dict) in enumerate(idx_masks):
            wp = obs.copy()
            dims = []
            for (k, v) in idx_dict.items():
                if v >= 0:
                    wp[k] = goal[v]
                    dims.append(k)
            for (k, v) in idx_dict.items():
                if v < 0:
                    wp[k] = wp[-v - 10]
                    dims.append(k)
                    dims.append(-v - 10)
            other_dims = [d for d in np.arange(len(wp)) if d not in dims]
            if other_dims_random:
                wp[other_dims] = env.observation_space.spaces['state_achieved_goal'].sample()[other_dims]
            list_of_waypoints[mask_id][i] = wp

        goals[i] = goal

    print("Done. Time:", time.time() - t1)

    return list_of_waypoints, goals


def infer_masks(env, idx_masks, mask_inference_variant):
    n = int(mask_inference_variant.get('n', 50))
    obs_noise = mask_inference_variant['noise']
    max_cond_num = mask_inference_variant['max_cond_num']
    normalize_sigma_inv = mask_inference_variant.get('normalize_sigma_inv', True)
    other_dims_random = mask_inference_variant.get('other_dims_random', True)

    list_of_waypoints, goals = gen_data(env, idx_masks, n, other_dims_random)

    # add noise to all of the data
    list_of_waypoints += np.random.normal(0, obs_noise, list_of_waypoints.shape)
    goals += np.random.normal(0, obs_noise, goals.shape)

    masks = {
        'mask_mu_w': [],
        'mask_mu_g': [],
        'mask_mu_mat': [],
        'mask_sigma_inv': [],
    }
    for (mask_id, waypoints) in enumerate(list_of_waypoints):
        mu = np.mean(np.concatenate((waypoints, goals), axis=1), axis=0)
        sigma = np.cov(np.concatenate((waypoints, goals), axis=1).T)
        mu_w, mu_g, mu_mat, sigma_inv = get_cond_distr_params(
            mu, sigma,
            x_dim=goals.shape[1],
            max_cond_num=max_cond_num
        )
        if normalize_sigma_inv:
            sigma_inv = sigma_inv / np.max(np.abs(sigma_inv))
        masks['mask_mu_w'].append(mu_w)
        masks['mask_mu_g'].append(mu_g)
        masks['mask_mu_mat'].append(mu_mat)
        masks['mask_sigma_inv'].append(sigma_inv)

    for k in masks.keys():
        masks[k] = np.array(masks[k])

    for mask_id in range(len(idx_masks)):
        # print('mask_mu_mat')
        # print_matrix(masks['mask_mu_mat'][mask_id])
        print('mask_sigma_inv for mask_id={}'.format(mask_id))
        print_matrix(masks['mask_sigma_inv'][mask_id])
    # exit()

    return masks