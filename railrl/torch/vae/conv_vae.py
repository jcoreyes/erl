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

from railrl.misc.ml_util import ConstantSchedule
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
from railrl.torch.core import PyTorchModule

e = MultitaskImagePoint2DEnv(render_size=84, render_onscreen=False, ball_radius=1)

class ConvVAETrainer():
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            imsize=84,
            lr=1e-3,
            do_scatterplot=False,
            normalize=False,
    ):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None:
            self.beta_schedule = ConstantSchedule(beta)
        self.imsize = imsize
        self.do_scatterplot = do_scatterplot

        """
        I think it's a bit nicer if the caller makes this call, i.e.
        ```
        m = ConvVAE(representation_size)
        if ptu.gpu_enabled():
            m.cuda()
        t = ConvVAETrainer(train_data, test_data, m)
        ```
        However, I'll leave this here for backwards-compatibility.
        """
        if ptu.gpu_enabled():
            model.cuda()

        self.model = model
        self.representation_size = model.representation_size
        self.input_channels = model.input_channels
        self.imlength = model.imlength

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.normalize = normalize

        if self.normalize:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)
            # self.train_dataset = ((self.train_dataset - self.train_data_mean)) + 1 / 2
            # self.test_dataset = ((self.test_dataset - self.train_data_mean)) + 1 / 2

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        if self.normalize:
            samples = ((samples - self.train_data_mean)) + 1 / 2
        return ptu.np_to_var(samples)

    def logprob(self, recon_x, x, mu, logvar):
        # Divide by batch_size rather than setting size_average=True because
        # otherwise the averaging will also happen across dimension 1 (the
        # pixels)
        return F.binary_cross_entropy(
            recon_x,
            x.narrow(start=0, length=self.imlength, dimension=1).contiguous().view(-1, self.imlength),
            size_average=False,
        ) / self.batch_size

    def kl_divergence(self, recon_x, x, mu, logvar):
        return - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        bces = []
        kles = []
        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(100):
            data = self.get_batch()
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            bce = self.logprob(recon_batch, data, mu, logvar)
            kle = self.kl_divergence(recon_batch, data, mu, logvar)
            loss = bce + beta * kle
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
        logger.record_tabular("train/BCE", np.mean(bces))
        logger.record_tabular("train/KL", np.mean(kles))
        logger.record_tabular("train/loss", np.mean(losses))

    def test_epoch(
            self,
            epoch,
            save_reconstruction=True,
            save_scatterplot=True,
    ):
        self.model.eval()
        losses = []
        bces = []
        kles = []
        zs = []
        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(10):
            data = self.get_batch(train=False)
            recon_batch, mu, logvar = self.model(data)
            bce = self.logprob(recon_batch, data, mu, logvar)
            kle = self.kl_divergence(recon_batch, data, mu, logvar)
            loss = bce + beta * kle

            z_data = ptu.get_numpy(mu.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.data[0])
            bces.append(bce.data[0])
            kles.append(kle.data[0])

            if batch_idx == 0 and save_reconstruction:
                n = min(data.size(0), 8)
                comparison = torch.cat([
                    data[:n].narrow(start=0, length=self.imlength, dimension=1)
                    .contiguous().view(
                        -1, self.input_channels, self.imsize, self.imsize
                    ),
                    recon_batch.view(
                        self.batch_size,
                        self.input_channels,
                        self.imsize,
                        self.imsize,
                    )[:n]
                ])
                save_dir = osp.join(logger.get_snapshot_dir(), 'r%d.png' % epoch)
                save_image(comparison.data.cpu(), save_dir, nrow=n)

        zs = np.array(zs)
        self.model.dist_mu = zs.mean(axis=0)
        self.model.dist_std = zs.std(axis=0)
        if self.do_scatterplot and save_scatterplot:
            self.plot_scattered(np.array(zs), epoch)

        logger.record_tabular("test/BCE", np.mean(bces))
        logger.record_tabular("test/KL", np.mean(kles))
        logger.record_tabular("test/loss", np.mean(losses))
        logger.record_tabular("beta", beta)
        logger.dump_tabular()

        logger.save_itr_params(epoch, self.model) # slow...
        logdir = logger.get_snapshot_dir()
        filename = osp.join(logdir, 'params.pkl')
        # torch.save(self.model, filename)

    def dump_samples(self, epoch):
        self.model.eval()
        sample = ptu.Variable(torch.randn(64, self.representation_size))
        sample = self.model.decode(sample).cpu()
        save_dir = osp.join(logger.get_snapshot_dir(), 's%d.png' % epoch)
        save_image(
            sample.data.view(64, self.input_channels, self.imsize, self.imsize),
            save_dir
        )

    def plot_scattered(self, z, epoch):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.log(__file__ + ": Unable to load matplotlib. Consider "
                                  "setting do_scatterplot to False")
            return
        dim_and_stds = [(i, np.std(z[:, i])) for i in range(z.shape[1])]
        dim_and_stds = sorted(
            dim_and_stds,
            key=lambda x: x[1]
        )
        dim1 = dim_and_stds[-1][0]
        dim2 = dim_and_stds[-2][0]
        plt.figure(figsize=(8, 8))
        plt.scatter(z[:, dim1], z[:, dim2], marker='o', edgecolor='none')
        if self.model.dist_mu is not None:
            x1 = self.model.dist_mu[dim1:dim1+1]
            y1 = self.model.dist_mu[dim2:dim2+1]
            x2 = self.model.dist_mu[dim1:dim1+1] + self.model.dist_std[dim1:dim1+1]
            y2 = self.model.dist_mu[dim2:dim2+1] + self.model.dist_std[dim2:dim2+1]
        plt.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
        axes = plt.gca()
        axes.set_xlim([-6, 6])
        axes.set_ylim([-6, 6])
        axes.set_title('dim {} vs dim {}'.format(dim1, dim2))
        plt.grid(True)
        save_file = osp.join(logger.get_snapshot_dir(), 'scatter%d.png' % epoch)
        plt.savefig(save_file)

# class ConvVAE(nn.Module):
class ConvVAE(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=84,
            added_fc_size=0,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            min_variance=1e-4,
            use_min_variance=True,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize**2 * self.input_channels
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.dist_mu = None
        self.dist_std = None

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.added_fc_size = added_fc_size
        self.init_w = init_w

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn3 = nn.BatchNorm2d(32)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 128 # kernel 3

        #self.hidden = nn.Linear(self.conv_output_dim + added_fc_size, representation_size)

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)

        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)

        self.fc4 = nn.Linear(self.conv_output_dim, imsize*imsize)
        self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=6, stride=3)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        # self.hidden_init(self.conv4.weight)
        # self.conv4.bias.data.fill_(0)
        # self.hidden_init(self.conv5.weight)
        # self.conv5.bias.data.fill_(0)
        # self.hidden_init(self.conv6.weight)
        # self.conv6.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)
        # self.hidden_init(self.fc3.weight)
        # self.fc3.bias.data.fill_(0)
        # self.fc3.weight.data.uniform_(-init_w, init_w)
        # self.fc3.bias.data.uniform_(-init_w, init_w)
        # self.hidden_init(self.fc4.weight)
        # self.fc4.bias.data.fill_(0)
        # self.fc4.weight.data.uniform_(-init_w, init_w)
        # self.fc4.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dimension=1)

        # batch_size = input.size(0)
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize, self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        h = x.view(-1, 128) # flatten
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dimension=1)
            h = torch.cat((h, fc_input), dim=1)
        #h = F.relu(self.hidden(h))
        mu = self.output_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.output_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = ptu.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, 32, 2, 2)
        x = F.relu(self.conv4(h))
        x = F.relu(self.conv5(x))
        x = self.conv6(x).view(-1, self.imsize*self.imsize*self.input_channels)
        return self.sigmoid(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def __getstate__(self):
        d = super().__getstate__()
        # Add these explicitly in case they were modified
        d["_dist_mu"] = self.dist_mu
        d["_dist_std"] = self.dist_std
        return d

    def __setstate__(self, d):
        super().__setstate__(d)
        self.dist_mu = d["_dist_mu"]
        self.dist_std = d["_dist_std"]

#class SpatialVAE(nn.Module):
class SpatialVAE(ConvVAE):
    def __init__(
            self,
            representation_size,
            num_feat_points,
            *args,
            temperature=1.0,
            **kwargs
    ):
#        self.save_init_params(locals())
        super().__init__(representation_size, *args, **kwargs)
        self.num_feat_points = num_feat_points
        self.conv3 = nn.Conv2d(32, self.num_feat_points, kernel_size=5, stride=3)
#        self.bn3 = nn.BatchNorm2d(32)

        test_mat = Variable(torch.zeros(1, self.input_channels, self.imsize, self.imsize))
        test_mat = self.conv1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.conv3(test_mat)
        self.out_size = int(np.prod(test_mat.shape))

        self.spatial_fc = nn.Linear(2 * self.num_feat_points + self.added_fc_size, 64)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 128 # kernel 3

        self.fc1 = nn.Linear(64, representation_size)
        self.fc2 = nn.Linear(64, representation_size)

        self.init_weights(self.init_w)
        self.temperature = temperature

    def init_weights_spatial(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)

        self.hidden_init(self.spatial_fc.weight)
        self.spatial_fc.bias.data.fill_(0)
        self.spatial_fc.weight.data.uniform_(-init_w, init_w)
        self.spatial_fc.bias.data.uniform_(-init_w, init_w)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dimension=1)

        # batch_size = input.size(0)
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize, self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        d = int((self.out_size // self.num_feat_points)**(1/2))
        x = x.view(-1, self.num_feat_points, d*d)
        x = F.softmax(x / self.temperature, 2)
        x = x.view(-1, self.num_feat_points, d, d)

        maps_x = torch.sum(x, 2)
        maps_y = torch.sum(x, 3)

        weights = ptu.np_to_var(np.arange(d) / (d + 1))

        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)

        x = torch.cat([fp_x, fp_y], 1)
        h = x.view(-1, self.num_feat_points * 2)
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dimension=1)
            h = torch.cat((h, fc_input), dim=1)
        h = F.relu(self.spatial_fc(h))
        mu = self.output_activation(self.fc1(h))
        logvar = self.output_activation(self.fc2(h))
        return mu, logvar

if __name__ == "__main__":
    m = ConvVAE(2)
    for epoch in range(10):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)
