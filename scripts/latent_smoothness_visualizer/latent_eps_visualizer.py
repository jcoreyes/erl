import sys
from railrl.misc.asset_loader import sync_down
from railrl.misc.asset_loader import load_local_or_remote_file
import torch
import pickle
import numpy as np
from multiworld.core.image_env import ImageEnv
from railrl.torch import pytorch_util as ptu
import matplotlib
matplotlib.use('Agg')
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from railrl.data_management.dataset  import \
        TrajectoryDataset, ImageObservationDataset, InitialObservationDataset
from railrl.data_management.images import normalize_image, unnormalize_image
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_subset import SawyerMultiobjectEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in, sawyer_pusher_camera_upright_v2
x_var = 0.2
x_low = -x_var
x_high = x_var
y_low = 0.5
y_high = 0.7
t = 0.05

def load_env():
	env = SawyerMultiobjectEnv(
		fixed_start=True,
		fixed_colors=False,
		num_objects=1,
		object_meshes=None,
		preload_obj_dict=
		[{'color1': [1, 1, 1],
		'color2': [1, 1, 1]}],
		num_scene_objects=[1],
		maxlen=0.1,
		action_repeat=1,
		puck_goal_low=(x_low + 0.01, y_low + 0.01),
		puck_goal_high=(x_high - 0.01, y_high - 0.01),
		hand_goal_low=(x_low + 0.01, y_low + 0.01),
		hand_goal_high=(x_high - 0.01, y_high - 0.01),
		mocap_low=(x_low, y_low, 0.0),
		mocap_high=(x_high, y_high, 0.5),
		object_low=(x_low + 0.01, y_low + 0.01, 0.02),
		object_high=(x_high - 0.01, y_high - 0.01, 0.02),
		use_textures=False,
		init_camera=sawyer_init_camera_zoomed_in,
		cylinder_radius=0.05,
		)
	wrapped_env = ImageEnv(
				env,
				48,
				init_camera=sawyer_init_camera_zoomed_in,
				transpose=True,
				normalize=True,
				non_presampled_goal_img_is_garbage=False,
			)
	return wrapped_env


def load_model(model_file):
	if model_file[0] == "/":
		local_path = model_file
	else:
		local_path = sync_down(model_file)
	model = pickle.load(open(local_path, "rb"))
	#model = torch.load(local_path, map_location='cpu')
	print("loaded", local_path)
	model.to("cpu")
	return model


class LatentVisualizer:
	def __init__(self, path):
		self.vae = load_model(path)
		self.env = load_env()
		self.plot_heatmap()

	def get_latents(self, val_min, val_max,num=3):
		hand_x = np.unique(np.linspace(val_min[0], val_max[0], num=num))
		hand_y = np.unique(np.linspace(val_min[1], val_max[1], num=num))
		puck_x = np.unique(np.linspace(val_min[2], val_max[2], num=num))
		puck_y = np.unique(np.linspace(val_min[3], val_max[3], num=num))

		center_state = (val_min + val_max) / 2
		all_imgs = []

		dist_matrix = np.zeros((hand_x.shape[0], hand_y.shape[0], puck_x.shape[0], puck_y.shape[0]))
		self.env.reset()

		center, _ = self.get_latent(center_state)

		for a in range(hand_x.shape[0]):
			for b in range(hand_y.shape[0]):
				for c in range(puck_x.shape[0]):
					for d in range(puck_y.shape[0]):
						state = np.array([hand_x[a], hand_y[b], puck_x[c], puck_y[d]])
						latent_state, img = self.get_latent(state)
						dist = np.linalg.norm(latent_state - center)
						dist_matrix[a,b,c,d] = dist
						all_imgs.append(img)

		self.save_env_images(all_imgs, num)
		vals = [hand_x, hand_y, puck_x, puck_y]
		return vals, dist_matrix.squeeze()


	def save_env_images(self, all_imgs, num_row):
		all_imgs = torch.stack(all_imgs)
		save_image(
	        all_imgs.data,
	        "/home/ashvin/data/sasha/heatmaps/images.pdf",
	        nrow=num_row,
	    )


	def get_latent(self, state):
		goal = dict(state_desired_goal=state)
		self.env.set_to_goal(goal)

		obs = self.env._get_obs()
		img = ptu.from_numpy(obs['image_observation']).view(1, -1)
		latent_state = ptu.get_numpy(self.vae.encode(img, cont=True)[0]).flatten()
		return latent_state, ptu.from_numpy(obs['image_observation']).reshape(3, 48, 48).transpose(1, 2)


	def get_keys(self, val_min, val_max):
		all_names = ["Hand: X Coordinate", "Hand: Y Coordinate", "Puck: X Coordinate", "Puck: Y Coordinate"]
		keys = []
		axis_names = []
		for i in range(val_min.shape[0]):
			if val_min[i] != val_max[i]:
				keys.append(i)
				axis_names.append(all_names[i])

		assert len(keys) == 2
		return keys, axis_names


	def plot_heatmap(self):
		# val_min = np.array([-0.2, -0.2, -0.2, 0.5])
		# val_max = np.array([-0.2, -0.2, 0.2, 0.7])

		val_min = np.array([0.0, 0.6, -0.2, 0.5])
		val_max = np.array([0.0, 0.6, 0.2, 0.7])

		keys, axis_names = self.get_keys(val_min, val_max)
		vals, dist = self.get_latents(val_min, val_max, num=20)

		plot = plt.figure()
		plt.contourf(vals[keys[0]], vals[keys[1]], dist, 10)
		plt.colorbar()
		plt.title("Latent Distance Visualizer", fontweight="bold")
		plt.xlabel(axis_names[0])
		plt.ylabel(axis_names[1])
		plt.savefig("/home/ashvin/data/sasha/heatmaps/pusher.pdf")




if __name__ == "__main__":
	#model_path = "/home/ashvin/data/sasha/testing/vqvae/vqvae-vqvae/run100/id0/vae.pkl"
	#model_path = "/home/ashvin/data/rail-khazatsky/sasha/testing/vae/vae-sparse/sasha/testing/vae/vae-sparse/run1/id1/vae.pkl"
	model_path = "/home/ashvin/data/rail-khazatsky/sasha/testing/vqvae/vqvae-vqvae/sasha/testing/vqvae/vqvae-vqvae/run57/id4/vae.pkl"
	LatentVisualizer(model_path)