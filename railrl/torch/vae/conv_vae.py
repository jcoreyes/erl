# Adapted from pytorch examples

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from railrl.policies.base import Policy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.modules import SelfOuterProductLinear, LayerNorm

from railrl.core import logger
import os.path as osp

from railrl.torch.vae.vae import VAE
import numpy as np

from railrl.envs.multitask.point2d import MultitaskImagePoint2DEnv
import numpy as np

e = MultitaskImagePoint2DEnv(render_size=84, render_onscreen=False, ball_radius=1)

def get_data(N = 10000):
    train_dataset = np.load("/tmp/train_dataset.npy")
    test_dataset = np.load("/tmp/test_dataset.npy")
    print("loaded data from saved files")
    return train_dataset, test_dataset

    # if not cached
    train_dataset = np.zeros((N, 84*84))
    for i in range(N):
        train_dataset[i, :] = e.reset()
    print("done making training data")

    test_dataset = np.zeros((N, 84*84))
    for i in range(N):
        test_dataset[i, :] = e.reset()
    print("done making test data")

    np.save("/tmp/train_dataset.npy", train_dataset)
    np.save("/tmp/test_dataset.npy", test_dataset)

    return train_dataset, test_dataset

class ConvVAE(nn.Module):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            batch_size=128, log_interval=0, use_cuda=False, beta=0.5
    ):
        super().__init__()
        self.log_interval = log_interval
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.beta = beta

        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(32)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 128 # kernel 3

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)

        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)

        self.fc4 = nn.Linear(self.conv_output_dim, 84*84)
        self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=6, stride=3)

        torch.nn.ConvTranspose2d

        self.init_weights(init_w)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.train_dataset, self.test_data = get_data()

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        # batch_size = input.size(0)
        x = input.view(-1, self.input_channels, 84, 84)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        h = x.view(-1, 128) # flatten
        mu = self.output_activation(self.fc1(h))
        logvar = self.output_activation(self.fc2(h))
        return mu, logvar

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        return ptu.from_numpy(dataset[ind, :])

    def logprob(self, recon_x, x, mu, logvar):
        return F.binary_cross_entropy(recon_x, x.view(-1, 84*84), size_average=False)

    def kl_divergence(self, recon_x, x, mu, logvar):
        return -self.beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, 32, 2, 2)
        x = F.relu(self.conv4(h))
        x = F.relu(self.conv5(x))
        x = self.conv6(x).view(-1, 84*84)
        return self.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 84*84))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train_epoch(self, epoch):
        self.train()
        losses = []
        bces = []
        kles = []
        for batch_idx in range(100):
            data = self.get_batch()
            data = Variable(data)
            if self.use_cuda:
                data = data.cuda()
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(data)
            bce = self.logprob(recon_batch, data, mu, logvar)
            kle = self.kl_divergence(recon_batch, data, mu, logvar)
            loss = bce + kle
            loss.backward()

            losses.append(loss.data[0])
            bces.append(bce.data[0])
            kles.append(kle.data[0])

            self.optimizer.step()
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.data[0] / len(data)))

        logger.record_tabular("train/epoch", epoch)
        logger.record_tabular("train/BCE", np.mean(bces) / self.batch_size)
        logger.record_tabular("train/KL", np.mean(kles) / self.batch_size)
        logger.record_tabular("train/loss", np.mean(losses) / self.batch_size)


    def test_epoch(self, epoch):
        self.eval()
        losses = []
        bces = []
        kles = []
        zs = []
        for batch_idx in range(10):
            data = self.get_batch()
            if self.use_cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self(data)
            bce = self.logprob(recon_batch, data, mu, logvar)
            kle = self.kl_divergence(recon_batch, data, mu, logvar)
            loss = bce + kle

            z_data = ptu.get_numpy(mu)
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.data[0])
            bces.append(bce.data[0])
            kles.append(kle.data[0])

            if batch_idx == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n].view(-1, 1, 84, 84),
                                      recon_batch.view(self.batch_size, 1, 84, 84)[:n]])
                save_dir = osp.join(logger.get_snapshot_dir(), 'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

        self.plot_scattered(np.array(zs), epoch)

        logger.record_tabular("test/BCE", np.mean(bces) / self.batch_size)
        logger.record_tabular("test/KL", np.mean(kles) / self.batch_size)
        logger.record_tabular("test/loss", np.mean(losses) / self.batch_size)
        logger.dump_tabular()

        logger.save_itr_params(epoch, self)

    def dump_samples(self, epoch):
        sample = Variable(torch.randn(64, self.representation_size))
        if self.use_cuda:
            sample = sample.cuda()
        sample = self.decode(sample).cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_image(sample.data.view(64, 1, 84, 84), save_dir)

    def plot_scattered(self, z, epoch):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, 0], z[:, 1], marker='o', edgecolor='none')
        axes = plt.gca()
        axes.set_xlim([-6, 6])
        axes.set_ylim([-6, 6])
        plt.grid(True)
        save_file = osp.join(logger.get_snapshot_dir(), 'scatter%d.png' % epoch)
        plt.savefig(save_file)

if __name__ == "__main__":
    m = ConvVAE(2)
    for epoch in range(10):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)
