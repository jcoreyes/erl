import torch
import numpy as np
import pickle
from rlkit.misc.asset_loader import load_local_or_remote_file
import rlkit.torch.pytorch_util as ptu

from torchvision.utils import save_image
from rlkit.data_management.images import normalize_image
import matplotlib.pyplot as plt


dataset_path = "/tmp/SawyerMultiobjectEnv_N5000_sawyer_init_camera_zoomed_in_imsize48_random_oracle_split_0.npy"
cvae_path = "/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/DCVAE/run103/id0/vae.pkl"
vae_path = "/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/baseline/run1/id0/itr_300.pkl"
prefix = "pusher1_"

# dataset_path = "/tmp/Multiobj2DEnv_N100000_sawyer_init_camera_zoomed_in_imsize48_random_oracle_split_0.npy"
# cvae_path = "/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/dynamics-cvae/run1/id0/itr_500.pkl"
# vae_path = "/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/baseline/run4/id0/itr_300.pkl"
# prefix = "pointmass1_"

N_ROWS = 3




dataset = load_local_or_remote_file(dataset_path)
dataset = dataset.item()

imlength = 6912
imsize = 48

N = dataset['observations'].shape[0]
test_p = 0.9
t = 0 #int(test_p * N)
n = 50
cvae = load_local_or_remote_file(cvae_path)
cvae.eval()
model = cvae.cpu()



cvae_distances = np.zeros((N - t, n))
for j in range(t, N):
    traj  = dataset['observations'][j, :, :] / 255.0
    n = traj.shape[0]

    x0 = traj[0, :] #dataset['env'][j, :]
    x0 = ptu.from_numpy(x0.reshape(1, -1))
    goal = traj[-1]
    latent_goal = model.encode(ptu.from_numpy(goal.reshape(1,-1)), x0, distrib=False)
    latent_goal = ptu.get_numpy(latent_goal)

    latents = model.encode(ptu.from_numpy(traj.reshape(n, imlength)), x0, distrib=False)
    latents = ptu.get_numpy(latents)
    latent_delta = latents - latent_goal
    for i in range(n):
        cvae_distances[j - t, i] = np.linalg.norm(latent_delta[i, :])

cvae_distances = cvae_distances.mean(axis=0) / np.amax(cvae_distances.mean(axis=0))


vae = load_local_or_remote_file(vae_path).to("cpu")
vae.eval()
model = vae.cpu()


vae_distances = np.zeros((N - t, n))
for j in range(t, N):
    traj  = dataset['observations'][j, :, :] / 255.0
    n = traj.shape[0]
    goal = traj[-1]
    latent_goal = model.encode(ptu.from_numpy(goal.reshape(1,-1)))[0]
    latent_goal = ptu.get_numpy(latent_goal)

    latents = model.encode(ptu.from_numpy(traj.reshape(n, imlength)))[0]
    latents = ptu.get_numpy(latents)
    latent_delta = latents - latent_goal
    for i in range(n):
        vae_distances[j - t, i] = np.linalg.norm(latent_delta[i, :])

vae_distances = vae_distances.mean(axis=0) / np.amax(vae_distances.mean(axis=0))





plt.plot(np.arange(n), vae_distances)
plt.plot(np.arange(n), cvae_distances)

plt.savefig("/home/khazatsky/rail/data/%sreward_curve.pdf" % prefix)
