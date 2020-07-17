import numpy as np
from scipy import linalg

def infer_masks(dataset, mask_inference_variant):
    obs_noise = mask_inference_variant['noise']
    max_cond_num = mask_inference_variant['max_cond_num']
    normalize_sigma_inv = mask_inference_variant.get('normalize_sigma_inv', True)
    sigma_inv_entry_threshold = mask_inference_variant.get('sigma_inv_entry_threshold', None)

    list_of_waypoints = dataset['list_of_waypoints']
    goals = dataset['goals']

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

        if sigma_inv_entry_threshold is not None:
            for i in range(len(sigma_inv)):
                for j in range(len(sigma_inv)):
                    if sigma_inv[i][j] / np.max(np.abs(sigma_inv)) <= sigma_inv_entry_threshold:
                        sigma_inv[i][j] = 0.0

        masks['mask_mu_w'].append(mu_w)
        masks['mask_mu_g'].append(mu_g)
        masks['mask_mu_mat'].append(mu_mat)
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