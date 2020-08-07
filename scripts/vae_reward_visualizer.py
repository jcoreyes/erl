import torch
import numpy as np
import pickle
from rlkit.misc.asset_loader import load_local_or_remote_file
import rlkit.torch.pytorch_util as ptu


vae_path = '/home/khazatsky/rail/data/rail-khazatsky/sasha/PCVAE/DCVAE/run20/id0/itr_600.pkl'

# vae_path = '/home/shikharbahl/research/rlkit-private/data/local/shikhar/corl2019/pointmass/real/run0/id0/vae.pkl'


vae = load_local_or_remote_file(vae_path)



dataset_path = '/home/khazatsky/rail/data/train_data.npy'
dataset = load_local_or_remote_file(dataset_path).item()

import matplotlib.pyplot as plt 

traj  = dataset['observations'][17]
n = traj.shape[0]

x0 = traj[0]
x0 = ptu.from_numpy(x0.reshape(1, -1))
goal = traj[-1]
vae = vae.cpu()
latent_goal = vae.encode(ptu.from_numpy(goal.reshape(1,-1)), x0, distrib=False)
decoded_goal, _ = vae.decode(latent_goal,x0)

log_probs = []
distances = []
for i in range(n): 
	x = traj[i]
	latent = vae.encode(ptu.from_numpy(x.reshape(1,-1)), x0, distrib=False)
	decoded, _ = vae.decode(latent, x0)
	distances.append(np.linalg.norm(ptu.get_numpy(latent) - ptu.get_numpy(latent_goal)))
	log_probs.append(ptu.get_numpy(vae.logprob(decoded_goal, decoded, mean=False).exp())[0])
plt.plot(np.arange(n), np.array(distances))
'''
dataset_path = '/home/shikharbahl/research/visual_foresight/examples/train_data.npy'
dataset = np.load(dataset_path).item()
traj  = dataset['observations'][0]
n = traj.shape[0]
import matplotlib.pyplot as plt

def get_distances(i):
	global vae
	traj  = dataset['observations'][i]
	x0 = traj[0]
	x0 = ptu.from_numpy(x0.reshape(1, -1))
	goal = traj[-1]
	vae = vae.cpu()
	latent_goal = vae.encode(ptu.from_numpy(goal.reshape(1,-1)), x0, distrib=False)
	decoded_goal, _ = vae.decode(latent_goal)

	n = traj.shape[0]
	log_probs = []
	distances = []
	for i in range(n):
		x = traj[i]
		latent = vae.encode(ptu.from_numpy(x.reshape(1,-1)), x0, distrib=False)
		decoded, _ = vae.decode(latent)
		distances.append(np.linalg.norm(ptu.get_numpy(latent) - ptu.get_numpy(latent_goal)))
		log_probs.append(ptu.get_numpy(vae.logprob(decoded_goal, decoded, mean=False).exp())[0])
	return np.array(distances)

dists = np.array([get_distances(i) for i in range(1)])
# import ipdb; ipdb.set_trace()
plt.plot(np.arange(n), np.mean(dists, axis=0))
'''
plt.show()
