import numpy as np
import torch

class Dataset:
	
	def __init__(self, train_data, test_data, info=None):
		self.train_data = train_data
		self.test_data = test_data
		self.info = info

	def get_batch(self, batch_size, test_data=False):
		if test_data:
			return np.random.choice(self.test_data, batch_size, replace=False)
		return np.random.choice(self.train_data, batch_size, replace=False)


class TrajectoryDataset(Dataset):
	
	def __init__(self, train_trajectories, test_trajectories, info=None):
		self.train_size = train_trajectories['obs'].shape[0]
		self.test_size = test_trajectories['obs'].shape[0]
		self.traj_length = test_trajectories['obs'].shape[1]
		self.info = info
		self.train_data = {
			'obs': train_trajectories['obs'],
			'acts': train_trajectories['acts']
			}

		self.test_data = {
			'obs': test_trajectories['obs'],
			'acts': test_trajectories['acts']
			}

	def get_batch(self, batch_size, test_data=False):
		if test_data:
			indicies = np.random.choice(np.arange(self.test_size), batch_size, replace=False)
			data_dict = {
				'obs': test_trajectories['obs'][indicies],
				'acts': test_trajectories['acts'][indicies]
				}
		else:
			indicies = np.random.choice(np.arange(self.train_size), batch_size, replace=False)
			data_dict = {
				'obs': train_trajectories['obs'][indicies],
				'acts': train_trajectories['acts'][indicies]
				}
		return data_dict

	def sample_trajectories(self, batch_size):
		#Assumes all trajectories are same length!


		traj_i = np.random.choice(np.arange(self.train_size), batch_size)
		trans_i = np.random.choice(np.arange(self.traj_length - 1), batch_size)
		data_dict = {
			'obs': self.train_data['obs'][traj_i][1][trans_i],
			'next_obs': self.train_data['obs'][traj_i][1][trans_i + 1],
			'actions': self.train_data['acts'][traj_i][1][trans_i]
			}
		return data_dict