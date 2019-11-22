"""Usage: first download the data (about 12.5GB).
pip install gdown
gdown https://drive.google.com/uc?id=16xlZWvR9Ml0TM0VEgrBGnFbcpzA4uLOY
tar -xvf bair_dataset_numpy.tar.gz

In each numpy file there are the following keys:
data = np.load("train00.npz")
images = data["images"] # T x K x H x 64 x 64 x 3, dtype=np.uint8
actions = data["actions"] # T x H x 4
endeff = data["endeff"] # T x H x 3

where:
T = # of trajectories,
K = camera ID (0 is top, 1 is front),
H = horizon = 15
"""
import numpy as np
import itertools
import torch

from railrl.data_management.images import normalize_image, unnormalize_image
from railrl.torch.core import np_to_pytorch_batch
import railrl.torch.pytorch_util as ptu

from torch.utils import data
from torchvision.transforms import ColorJitter, RandomResizedCrop
from PIL import Image

import torchvision.transforms.functional as F

from railrl.torch.data import BatchLoader, InfiniteBatchLoader
from railrl.data_management.external.bair_dataset.config import BAIR_DATASET_LOCATION

class BAIRDataset(data.Dataset):
    def __init__(self, is_train, camera=1, info=None):
        self.is_train = is_train
        self.N = 1024 if is_train else 256
        self.traj_length = 15
        self.camera = camera

        suffix = "train00.npz" if is_train else "test.npz"
        filename = BAIR_DATASET_LOCATION + suffix
        self.data = np.load(filename)
        self.images = self.data['images']
        print("loaded BAIR dataset:", self.images.shape)

        # self.jitter = ColorJitter((0.7,1.3), (0.95,1.05), (0.95,1.05), (-0.05,0.05))
        # self.crop = RandomResizedCrop((48, 48), (0.9, 0.9), (1, 1))
        # RandomResizedCrop((int(sqrt(self.imlength)), int(sqrt(self.imlength))), (0.9, 0.9), (1, 1))

    def __len__(self):
        return self.N * self.traj_length

    def __getitem__(self, idx):
        traj_i = idx // self.traj_length
        trans_i = idx % self.traj_length

        # x = Image.fromarray(self.data['observations'][traj_i, trans_i].reshape(48, 48, 3), mode='RGB')
        # c = Image.fromarray(self.data['env'][traj_i].reshape(48, 48, 3), mode='RGB')

        # # upsampling gives bad images so random resizing params set to 1 for now
        # # crop = self.crop.get_params(c, (0.9, 0.9), (1, 1))
        # crop = self.crop.get_params(c, (1, 1), (1, 1))
        # jitter = self.jitter.get_params((0.7,1.3), (0.95,1.05), (0.95,1.05), (-0.05,0.05))
        # # jitter = self.jitter.get_params((0.5,1.5), (0.9,1.1), (0.9,1.1), (-0.1,0.1))
        # # jitter = self.jitter.get_params(0.5, 0.1, 0.1, 0.1)

        # x = jitter(F.resized_crop(x, crop[0], crop[1], crop[2], crop[3], (48, 48), Image.BICUBIC))
        # c = jitter(F.resized_crop(c, crop[0], crop[1], crop[2], crop[3], (48, 48), Image.BICUBIC))
        # x_t = normalize_image(np.array(x).flatten()).squeeze()
        # env = normalize_image(np.array(c).flatten()).squeeze()

        data_dict = {
            'x_t': self.images[traj_i, self.camera, trans_i, 8:56, 8:56, :].transpose().flatten() / 255.0,
            'env': self.images[traj_i, self.camera, 0, 8:56, 8:56, :].transpose().flatten() / 255.0,
        }
        return data_dict


def generate_dataset(variant):
    train_dataset = BAIRDataset(is_train=True)
    test_dataset = BAIRDataset(is_train=False)

    train_batch_loader_kwargs = variant.get(
        'train_batch_loader_kwargs',
        dict(batch_size=32, num_workers=0, )
    )
    test_batch_loader_kwargs = variant.get(
        'test_batch_loader_kwargs',
        dict(batch_size=32, num_workers=0, )
    )

    train_data_loader = data.DataLoader(train_dataset,
        shuffle=True, drop_last=True, **train_batch_loader_kwargs)
    test_data_loader = data.DataLoader(test_dataset,
        shuffle=True, drop_last=True, **test_batch_loader_kwargs)

    train_dataset = InfiniteBatchLoader(train_data_loader)
    test_dataset = InfiniteBatchLoader(test_data_loader)
    info = {}

    return train_dataset, test_dataset, info
