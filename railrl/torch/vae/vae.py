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

from railrl.core import logger
import os.path as osp

import numpy as np

class VAE(nn.Module):
    def __init__(self, batch_size=128, log_interval=0, use_cuda=False, beta=0.5):
        super(VAE, self).__init__()

        self.setup_network()
        self.log_interval = log_interval
        self.use_cuda = use_cuda
        self.batch_size = batch_size
        self.beta = beta

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.get_data(self.batch_size) # initialize data loaders

    def setup_network(self):
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def get_data(self, batch_size, **kwargs):
        """Loads MNIST data for testing"""
        # kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('/tmp/data', train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=True, **kwargs)

    def logprob(self, recon_x, x, mu, logvar):
        return F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    def kl_divergence(self, recon_x, x, mu, logvar):
        return -self.beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def train_epoch(self, epoch):
        self.train()
        losses = []
        bces = []
        kles = []
        for batch_idx, (data, _) in enumerate(self.train_loader):
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
        for i, (data, _) in enumerate(self.test_loader):
            if self.use_cuda:
                data = data.cuda()
            data = Variable(data, requires_grad=True)
            recon_batch, mu, logvar = self(data)
            bce = self.logprob(recon_batch, data, mu, logvar)
            kle = self.kl_divergence(recon_batch, data, mu, logvar)
            loss = bce + kle

            losses.append(loss.data[0])
            bces.append(bce.data[0])
            kles.append(kle.data[0])

            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(self.batch_size, 1, 28, 28)[:n]])
                save_dir = osp.join(logger.get_snapshot_dir(), 'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

        logger.record_tabular("test/BCE", np.mean(bces) / self.batch_size)
        logger.record_tabular("test/KL", np.mean(kles) / self.batch_size)
        logger.record_tabular("test/loss", np.mean(losses) / self.batch_size)
        logger.dump_tabular()

    def dump_samples(self, epoch):
        sample = Variable(torch.randn(64, 20))
        if self.use_cuda:
            sample = sample.cuda()
        sample = self.decode(sample).cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_image(sample.data.view(64, 1, 28, 28), save_dir)

if __name__ == "__main__":
    m = VAE()
    for epoch in range(1, args.epochs + 1):
        m.train_epoch(epoch)
        m.test_epoch(epoch)

