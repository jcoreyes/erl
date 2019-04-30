import numpy as np
import torch

from railrl.data_management.images import normalize_image

from railrl.torch.core import np_to_pytorch_batch

class Dataset:
    def random_batch(self, batch_size):
        raise NotImplementedError


class ObservationDataset(Dataset):
    def __init__(self, data, info=None):
        self.data = data
        self.size = data.shape[0]
        self.info = info

    def random_batch(self, batch_size):
        i = np.random.choice(self.size, batch_size, replace=False)
        data_dict = {
            'observations': self.data[i, :],
        }
        return np_to_pytorch_batch(data_dict)


class ImageObservationDataset(Dataset):
    def __init__(self, data, normalize=True, info=None):
        assert data.dtype == np.uint8
        self.data = data
        self.size = data.shape[0]
        self.info = info
        self.normalize = normalize

    def random_batch(self, batch_size):
        i = np.random.choice(self.size, batch_size, replace=False)
        obs = self.data[i, :]
        if self.normalize:
            obs = normalize_image(obs)
        data_dict = {
            'observations': obs,
        }
        return np_to_pytorch_batch(data_dict)


class TrajectoryDataset(Dataset):
    def __init__(self, data, info=None):
        self.size = data['observations'].shape[0]
        self.traj_length = data['observations'].shape[1]
        self.data = data
        self.info = info

    def random_batch(self, batch_size):
        traj_i = np.random.choice(np.arange(self.size), batch_size)
        trans_i = np.random.choice(np.arange(self.traj_length - 1), batch_size)
        data_dict = {
            'observations': self.data['observations'][traj_i, trans_i, :],
            'next_observations': self.data['observations'][traj_i, trans_i + 1, :],
            'actions': self.data['actions'][traj_i, trans_i, :]
        }
        return np_to_pytorch_batch(data_dict)


class InitialObservationDataset(Dataset):
    def __init__(self, data, info=None):
        self.size = data['observations'].shape[0]
        self.traj_length = data['observations'].shape[1]
        self.data = data
        self.info = info

    def random_batch(self, batch_size):
        traj_i = np.random.choice(self.size, batch_size)
        trans_i = np.random.choice(self.traj_length, batch_size)

        x_0 = normalize_image(self.data['observations'][traj_i, 0, :])
        x_t = normalize_image(self.data['observations'][traj_i, trans_i, :])
        x = np.concatenate((x_t, x_0), 1)
        data_dict = {
            'observations': x,
            'x_t': x_t,
            'x_0': x_0,
        }
        return np_to_pytorch_batch(data_dict)


class TripletReplayBufferWrapper(Dataset):
    def __init__(self, replay_buffer, horizon, info=None):
        self.replay_buffer = replay_buffer
        self.horizon = horizon

    def random_batch(self, batch_size):
        num_traj = self.replay_buffer._size // self.horizon
        traj_i = np.random.choice(num_traj, batch_size)
        trans_i = np.random.choice(self.horizon - 2, batch_size)

        indices = traj_i * self.horizon + trans_i
        batch = dict(
            x0=self.replay_buffer._obs["image_observation"][indices],
            x1=self.replay_buffer._obs["image_observation"][indices+1],
            x2=self.replay_buffer._obs["image_observation"][indices+2],
        )
        return np_to_pytorch_batch(batch)
