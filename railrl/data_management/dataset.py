import numpy as np
import itertools
import torch

from railrl.data_management.images import normalize_image, unnormalize_image

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
        i = np.random.choice(self.size, batch_size, replace=(self.size < batch_size))
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
        i = np.random.choice(self.size, batch_size, replace=(self.size < batch_size))
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
            'actions': self.data['actions'][traj_i, trans_i, :],
        }

        return np_to_pytorch_batch(data_dict)


class EnvironmentDataset(Dataset):
    def __init__(self, data, info=None):
        self.num_envs = data['observations'].shape[0]
        self.sample_size = data['observations'].shape[1]
        self.data = data
        self.info = info

    def random_batch(self, batch_size):
        env_i = np.random.choice(self.num_envs, batch_size)
        trans_i = np.random.choice(self.sample_size, batch_size)

        match_i = np.random.choice(self.num_envs, batch_size)
        trans_x = np.random.choice(self.sample_size, batch_size)
        trans_y = np.random.choice(self.sample_size, batch_size)

        rand_a = np.random.choice(self.num_envs - 1, batch_size // 2)
        rand_b = np.add(rand_a, np.ones(batch_size // 2)).astype(int)

        trans_m = np.random.choice(self.sample_size, batch_size // 2)
        trans_n = np.random.choice(self.sample_size, batch_size // 2)

        matches = np.random.uniform(0, 0.1, batch_size // 2)
        nonmatches = np.random.uniform(0.9, 1, batch_size // 2)
        swap_count = int(batch_size * 0.05)

        matches[:swap_count], nonmatches[:swap_count] = nonmatches[:swap_count], matches[:swap_count]
        labels = np.concatenate([matches, nonmatches])

        data_dict = {
            'observations': self.data['observations'][env_i, trans_i, :],
            'env_set_1': self.data['observations'][match_i, trans_x, :],
            'env_set_2': self.data['observations'][match_i, trans_y, :],
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
        # conditioning = np.random.choice(self.traj_length, batch_size)
        # env = normalize_image(self.data['observations'][traj_i, conditioning, :])
        try:
            env = normalize_image(self.data['env'][traj_i, :])
        except:
            env = normalize_image(self.data['observations'][traj_i, 0, :])
        x_t = normalize_image(self.data['observations'][traj_i, trans_i, :])

        episode_num = np.random.randint(0, self.size)
        episode_obs = normalize_image(self.data['observations'][episode_num, :8, :])


        data_dict = {
            'x_t': x_t,
            'env': env,
            'episode_obs': episode_obs,
        }
        return np_to_pytorch_batch(data_dict)


class ConditionalDynamicsDataset(Dataset):
    def __init__(self, data, info=None):
        self.size = data['observations'].shape[0]
        self.traj_length = data['observations'].shape[1]
        self.data = data
        self.info = info

    def random_batch(self, batch_size):
        traj_i = np.random.choice(self.size, batch_size)
        trans_i = np.random.choice(self.traj_length - 1, batch_size)

        try:
            env = normalize_image(self.data['env'][traj_i, :])
        except:
            env = normalize_image(self.data['observations'][traj_i, 0, :])

        x_t = normalize_image(self.data['observations'][traj_i, trans_i, :])
        x_next = normalize_image(self.data['observations'][traj_i, trans_i + 1, :])

        episode_num = np.random.randint(0, self.size)
        episode_obs = normalize_image(self.data['observations'][episode_num, :8, :])

        data_dict = {
            'x_t': x_t,
            'x_next': x_next,
            'env': env,
            'actions': self.data['actions'][traj_i, trans_i, :],
            'episode_obs': episode_obs,
            'episode_acts': self.data['actions'][episode_num, :7, :],
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
