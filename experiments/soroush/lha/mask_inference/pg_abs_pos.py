from railrl.launchers.sets.mask_inference import (
    infer_masks,
    print_matrix,
)
import numpy as np

mask_inference_variant = dict(
    n=100,
    noise=0.10,
    max_cond_num=1e2,
    normalize_sigma_inv=True,
    sigma_inv_entry_threshold=0.10,
)
dataset_path = "/home/soroush/data/local/" \
               "07-17-pg-example-set/07-17-pg-example-set_2020_07_17_14_01_17_id000--s266/example_dataset.npy"

dataset = np.load(dataset_path)[()]
masks = infer_masks(dataset, mask_inference_variant)


# for i in range(num_subtasks):
#     waypoints = list_of_waypoints[:,i,:]
#
#     for j in range(1):
#         state = states[j]
#         goal = goals[j]
#         goal = goal.copy()
#         # goal[4:6] = goal[0:2]
#
#         if context_conditioned:
#             mu_w_given_c, sigma_w_given_c = get_cond_distr(mu, sigma, goal)
#         else:
#             mu_w_given_c, sigma_w_given_c = mu[:len(goal)], sigma[:len(goal),:len(goal)]
#
#         w, v = np.linalg.eig(sigma_w_given_c)
#         if j == 0:
#             print("eig:", sorted(w))
#             print("cond number:", np.max(w) / np.min(w))
#         l, h = np.min(w), np.max(w)
#         target = 1 / cond_num
#         # if l < target:
#         #     eps = target
#         # else:
#         #     eps = 0
#         if (l / h) < target:
#             eps = (h * target - l) / (1 - target)
#         else:
#             eps = 0
#         sigma_w_given_c += eps * np.identity(sigma_w_given_c.shape[0])
#
#         sigma_inv = linalg.inv(sigma_w_given_c)
#         # sigma_inv = sigma_inv / np.max(np.abs(sigma_inv))
#         # print(sigma_w_given_c)
#         if j == 0:
#             print_matrix(
#                 sigma_w_given_c,
#                 format="raw",
#                 normalize=True,
#                 threshold=0.4,
#                 precision=10,
#             )
#             print_matrix(
#                 # sigma_inv[[2, 3, 6]][:,[2, 3, 6]],
#                 sigma_inv,
#                 format="raw",
#                 normalize=True,
#                 threshold=0.4,
#                 precision=2,
#             )
#             # exit()
#
#         # print("s:", state)
#         # print("g:", goal)
#         # print("w:", mu_w_given_c)
#         # print()
#
#         if vis_distr:
#             if i == 0:
#                 list_of_dims = [[0, 1], [2, 3], [0, 2], [1, 3]]
#             elif i == 2:
#                 list_of_dims = [[2, len(goal) - 2]]
#             else:
#                 list_of_dims = [[2, 3], [4, 5], [2, 4], [3, 5]]
#             plot_Gaussian(
#                 mu_w_given_c,
#                 Sigma=sigma_w_given_c,
#                 pt1=goal, #pt2=state,
#                 list_of_dims=list_of_dims,
#             )


