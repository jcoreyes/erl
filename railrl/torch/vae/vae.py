# Adapted from pytorch examples

from __future__ import print_function
import copy
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from railrl.core import logger
import os.path as osp
import numpy as np
from railrl.misc.ml_util import ConstantSchedule
from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule
from railrl.torch.networks import Mlp, TwoHeadMlp
import railrl.torch.pytorch_util as ptu

class VAETrainer():
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            batch_size=128,
            log_interval=0,
            beta=0.5,
            beta_schedule=None,
            lr=1e-3,
            do_scatterplot=False,
            normalize=False,
            is_auto_encoder=False,
    ):
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.beta = beta
        if is_auto_encoder:
            self.beta = 0
        self.beta_schedule = beta_schedule
        if self.beta_schedule is None:
            self.beta_schedule = ConstantSchedule(self.beta)
        self.do_scatterplot = do_scatterplot

        if ptu.gpu_enabled():
            model.cuda()

        self.model = model
        self.representation_size = model.representation_size

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_dataset, self.test_dataset = train_dataset, test_dataset
        self.normalize = normalize
        self.mse = nn.MSELoss()

        if self.normalize:
            self.train_data_mean = np.mean(self.train_dataset, axis=0)

    def get_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        ind = np.random.randint(0, len(dataset), self.batch_size)
        samples = dataset[ind, :]
        if self.normalize:
            samples = ((samples - self.train_data_mean) + 1) / 2
        return ptu.np_to_var(samples)


    def get_debug_batch(self, train=True):
        dataset = self.train_dataset if train else self.test_dataset
        X, Y = dataset
        ind = np.random.randint(0, Y.shape[0], self.batch_size)
        X = X[ind, :]
        Y = Y[ind, :]
        return ptu.np_to_var(X), ptu.np_to_var(Y)

    def logprob(self, recon_x, x):
        return self.mse(recon_x, x)

    def kl_divergence(self, mu, logvar):
        kl = - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return kl

    def train_epoch(self, epoch, sample_batch=None, batches=100, from_rl=False):
        self.model.train()
        losses = []
        kles = []
        mses = []
        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(batches):
            if sample_batch is not None:
                data = sample_batch(self.batch_size)
            else:
                data = self.get_batch()
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            mse = self.logprob(recon_batch, data)
            kle = self.kl_divergence(mu, logvar)
            # print('Mu', mu.mean().data[0])
            # print('Logvar', logvar.mean().exp().data[0])
            # print('MSE', mse.data[0])
            loss = mse + beta * kle
            loss.backward()

            losses.append(loss.data[0])
            mses.append(mse.data[0])
            kles.append(kle.data[0])

            self.optimizer.step()
            if self.log_interval and batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.data[0] / len(data)))

        if not from_rl:
            logger.record_tabular("train/epoch", epoch)
            logger.record_tabular("train/MSE", np.mean(mses))
            logger.record_tabular("train/KL", np.mean(kles))
            logger.record_tabular("train/loss", np.mean(losses))

    def test_epoch(
            self,
            epoch,
            save_scatterplot=True,
            save_vae=True,
            from_rl=False,
    ):
        self.model.eval()
        losses = []
        kles = []
        zs = []
        mses = []
        beta = self.beta_schedule.get_value(epoch)
        for batch_idx in range(100):
            data = self.get_batch(train=False)
            recon_batch, mu, logvar = self.model(data)
            mse = self.logprob(recon_batch, data)
            kle = self.kl_divergence(mu, logvar)
            loss = mse + beta * kle
            z_data = ptu.get_numpy(mu.cpu())
            for i in range(len(z_data)):
                zs.append(z_data[i, :])
            losses.append(loss.data[0])
            mses.append(mse.data[0])
            kles.append(kle.data[0])
        zs = np.array(zs)
        self.model.dist_mu = zs.mean(axis=0)
        self.model.dist_std = zs.std(axis=0)
        if self.do_scatterplot and save_scatterplot:
            self.plot_scattered(np.array(zs), epoch)

        if not from_rl:
            logger.record_tabular("test/MSE", np.mean(mses))
            logger.record_tabular("test/KL", np.mean(kles))
            logger.record_tabular("test/loss", np.mean(losses))
            logger.record_tabular("beta", beta)
            logger.dump_tabular()
            if save_vae:
                logger.save_itr_params(epoch, self.model)  # slow...

    def dump_samples(self, epoch):
        pass

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

class VAE(PyTorchModule):
    def __init__(
            self,
            representation_size,
            input_size,
            hidden_sizes,
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
            output_scale=1,
            layer_norm=False,
    ):
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.init_w = init_w
        hidden_sizes = list(hidden_sizes)
        self.encoder=TwoHeadMlp(hidden_sizes, representation_size, representation_size, input_size, layer_norm=layer_norm)
        hidden_sizes.reverse()
        self.decoder=Mlp(hidden_sizes, input_size, representation_size, layer_norm=layer_norm, output_activation=output_activation, output_bias=None)
        self.output_scale = output_scale

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decoder(z) * self.output_scale

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        # TODO: is the deepcopy necessary?
        self.__dict__.update(copy.deepcopy(d))


class AutoEncoder(VAE):
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(z), mu, logvar