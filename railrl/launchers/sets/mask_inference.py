import numpy as np
from scipy import linalg

from railrl.launchers.sets.example_set_gen import gen_example_sets

def get_mask_params(
        env,
        mask_format,
        example_set_variant,
        mask_inference_variant,
):
    masks = {}
    goal_dim = env.observation_space.spaces['state_desired_goal'].low.size
    if mask_format == 'vector':
        mask_keys = ['mask']
        mask_dims = [(goal_dim,)]
    elif mask_format == 'matrix':
        mask_keys = ['mask']
        mask_dims = [(goal_dim, goal_dim)]
    elif mask_format == 'distribution':
        mask_keys = ['mask_mu', 'mask_sigma_inv']
        mask_dims = [(goal_dim,), (goal_dim, goal_dim)]
    elif mask_format == 'cond_distribution':
        mask_keys = ['mask_mu_w', 'mask_mu_g', 'mask_mu_mat', 'mask_sigma_inv']
        mask_dims = [(goal_dim,), (goal_dim,), (goal_dim, goal_dim), (goal_dim, goal_dim)]
    else:
        raise TypeError

    subtask_codes = example_set_variant['subtask_codes']
    num_masks = len(subtask_codes)
    for mask_key, mask_dim in zip(mask_keys, mask_dims):
        masks[mask_key] = np.zeros([num_masks] + list(mask_dim))

    infer_masks = mask_inference_variant['infer_masks']
    if infer_masks:
        example_dataset = gen_example_sets(env, example_set_variant)
    else:
        if mask_format == 'vector':
            mask_key = 'mask'
        elif mask_format == 'matrix':
            mask_key = 'mask'
        elif mask_format == 'distribution':
            mask_key = 'mask_sigma_inv'
        elif mask_format == 'cond_distribution':
            mask_key = 'mask_sigma_inv'
        else:
            raise TypeError

        for (mask_id, idx_dict) in enumerate(subtask_codes):
            for (k, v) in idx_dict.items():
                if mask_format == 'vector':
                    assert k == v
                    masks['mask'][mask_id][k] = 1
                elif mask_format in ['matrix', 'distribution', 'cond_distribution']:
                    if v >= 0:
                        assert k == v
                        masks[mask_key][mask_id][k, k] = 1
                    else:
                        src_idx = k
                        targ_idx = -(v + 10)
                        masks[mask_key][mask_id][src_idx, src_idx] = 1
                        masks[mask_key][mask_id][targ_idx, targ_idx] = 1
                        masks[mask_key][mask_id][src_idx, targ_idx] = -1
                        masks[mask_key][mask_id][targ_idx, src_idx] = -1

    # masks['mask_mu_mat'][:] = np.identity(masks['mask_mu_mat'].shape[-1])

    return masks

def infer_masks(
        dataset,
        noise,
        max_cond_num,
        mask_format,
        normalize_mask=True,
        mask_threshold=None,
):
    list_of_waypoints = dataset['list_of_waypoints']
    goals = dataset['goals']

    # add noise to all of the data
    list_of_waypoints += np.random.normal(0, noise, list_of_waypoints.shape)
    goals += np.random.normal(0, noise, goals.shape)

    if mask_format == 'cond_distribution':
        masks = {
            'mask_mu_w': [],
            'mask_mu_g': [],
            'mask_mu_mat': [],
            'mask_sigma_inv': [],
        }
    elif mask_format == 'distribution':
        masks = {
            'mask_mu': [],
            'mask_sigma_inv': [],
        }
    for (mask_id, waypoints) in enumerate(list_of_waypoints):
        if mask_format == 'cond_distribution':
            mu_w, mu_g, mu_mat, sigma = get_cond_distr_params(
                mu=np.mean(np.concatenate((waypoints, goals), axis=1), axis=0),
                sigma=np.cov(np.concatenate((waypoints, goals), axis=1).T),
                x_dim=goals.shape[1],
            )
        elif mask_format == 'distribution':
            mu = np.mean(waypoints, axis=0)
            sigma = np.cov(waypoints.T)

        w, v = np.linalg.eig(sigma)
        l, h = np.min(w), np.max(w)
        target = 1 / max_cond_num
        if (l / h) < target:
            eps = (h * target - l) / (1 - target)
        else:
            eps = 0
        sigma_inv = linalg.inv(sigma + eps * np.identity(sigma.shape[0]))

        if normalize_mask:
            sigma_inv = sigma_inv / np.max(np.abs(sigma_inv))

        if mask_threshold is not None:
            for i in range(len(sigma_inv)):
                for j in range(len(sigma_inv)):
                    if sigma_inv[i][j] / np.max(np.abs(sigma_inv)) <= mask_threshold:
                        sigma_inv[i][j] = 0.0

        if mask_format == 'cond_distribution':
            masks['mask_mu_w'].append(mu_w)
            masks['mask_mu_g'].append(mu_g)
            masks['mask_mu_mat'].append(mu_mat)
            masks['mask_sigma_inv'].append(sigma_inv)
        elif mask_format == 'distribution':
            masks['mask_mu'].append(mu)
            masks['mask_sigma_inv'].append(sigma_inv)

    for k in masks.keys():
        masks[k] = np.array(masks[k])

    for mask_id in range(len(list_of_waypoints)):
        # print('mask_mu_mat')
        # print_matrix(masks['mask_mu_mat'][mask_id])
        print('mask_sigma_inv for mask_id={}'.format(mask_id))
        print_matrix(masks['mask_sigma_inv'][mask_id], precision=5) #precision=5
        # print(masks['mask_sigma_inv'][mask_id].diagonal())
    # exit()

    return masks

def get_cond_distr_params(mu, sigma, x_dim):
    mu_x = mu[:x_dim]
    mu_y = mu[x_dim:]

    sigma_xx = sigma[:x_dim, :x_dim]
    sigma_yy = sigma[x_dim:, x_dim:]
    sigma_xy = sigma[:x_dim, x_dim:]
    sigma_yx = sigma[x_dim:, :x_dim]

    sigma_yy_inv = linalg.inv(sigma_yy)

    mu_mat = sigma_xy @ sigma_yy_inv
    sigma_xgy = sigma_xx - sigma_xy @ sigma_yy_inv @ sigma_yx

    return mu_x, mu_y, mu_mat, sigma_xgy

def print_matrix(matrix, format="raw", threshold=0.1, normalize=True, precision=5):
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
                if precision == 2:
                    print("{:.2f}".format(value), end=" ")
                elif precision == 5:
                    print("{:.5f}".format(value), end=" ")
        print()
    print()

def plot_Gaussian(
        mu,
        sigma=None,
        sigma_inv=None,
        bounds=None,
        list_of_dims=[[0, 1], [2, 3], [0, 2], [1, 3]],
        pt1=None,
        pt2=None
):
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    num_subplots = len(list_of_dims)
    if num_subplots == 1:
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig, axs = plt.subplots(2, num_subplots // 2, figsize=(10, 10))
    lb, ub = bounds
    gran = (ub - lb) / 50
    x, y = np.mgrid[lb:ub:gran, lb:ub:gran]
    pos = np.dstack((x, y))

    assert (sigma is not None) ^ (sigma_inv is not None)
    if sigma is None:
        sigma = linalg.inv(sigma_inv + np.eye(len(mu)) * 1e-6)

    for i in range(len(list_of_dims)):
        dims = list_of_dims[i]
        rv = multivariate_normal(mu[dims], sigma[dims][:,dims], allow_singular=True)

        if num_subplots == 1:
            axs_obj = axs
        else:
            plt_idx1 = i // 2
            plt_idx2 = i % 2
            axs_obj = axs[plt_idx1, plt_idx2]

        axs_obj.contourf(x, y, rv.logpdf(pos))
        axs_obj.set_title(str(dims))

        if pt1 is not None:
            axs_obj.scatter([pt1[dims][0]], [pt1[dims][1]])

        if pt2 is not None:
            axs_obj.scatter([pt2[dims][0]], [pt2[dims][1]])

    plt.show()