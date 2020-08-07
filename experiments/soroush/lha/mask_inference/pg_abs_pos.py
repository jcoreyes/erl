from rlkit.launchers.sets.mask_inference import (
    infer_masks,
    print_matrix,
    plot_Gaussian,
)
import numpy as np

n = 100
mask_inference_variant = dict(
    noise=0.10,
    max_cond_num=1e2,
    normalize_sigma_inv=True,
    sigma_inv_entry_threshold=0.10,
)
dataset_path = "/home/soroush/data/local/" \
               "07-17-pg-example-set/07-17-pg-example-set_2020_07_17_14_01_17_id000--s266/example_dataset.npy"
vis_distr = True

dataset = np.load(dataset_path)[()]
data_idxs = np.arange(dataset['list_of_waypoints'].shape[1])
np.random.shuffle(data_idxs)
data_idxs = data_idxs[:n]
list_of_waypoints = dataset['list_of_waypoints'][:, data_idxs]
goals = dataset['goals'][data_idxs]
dataset = {
    'list_of_waypoints': list_of_waypoints,
    'goals': goals,
}

masks = infer_masks(dataset, mask_inference_variant)

for i in [0]:
    distr_params = {key: masks[key][i] for key in masks.keys()}
    waypoints = list_of_waypoints[:,i,:]

    for j in range(1):
        goal = goals[j]
        goal = goal.copy()

        mu = distr_params['mask_mu_w'] + distr_params['mask_mu_mat'] @ (goal - distr_params['mask_mu_g'])
        sigma_inv =  distr_params['mask_sigma_inv']

        if j == 0:
            print_matrix(
                sigma_inv,
                format="raw",
                normalize=True,
                threshold=0.4,
                precision=2,
            )

        # print("s:", state)
        # print("g:", goal)
        # print("w:", mu_w_given_c)
        # print()

        if vis_distr:
            list_of_dims = [
                # [0, 1],
                # [2, 3],
                [0, 2],
                # [1, 3]
            ]
            plot_Gaussian(
                mu,
                sigma_inv=sigma_inv,
                bounds=[-4, 4],
                pt1=goal,
                list_of_dims=list_of_dims,
            )


