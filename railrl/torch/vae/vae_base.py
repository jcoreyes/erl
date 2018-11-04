import torch

from railrl.torch.core import PyTorchModule
import numpy as np
import abc

from railrl.torch.networks import CNN
from torch import nn


class VAEBase(PyTorchModule,  metaclass=abc.ABCMeta):
    def __init__(
            self,
            representation_size,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

    @abc.abstractmethod
    def encode(self, input):
        """
        :param input:
        :return: mu, logvar
        """
        raise NotImplementedError()

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    @abc.abstractmethod
    def decode(self, z):
        """
        :param z:
        :return: reconstruction
        """
        raise NotImplementedError()

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

class ConvVAE(VAEBase):
    def __init__(
            self,
            representation_size,
            conv_args,
            conv_kwargs,
            deconv_args,
            deconv_kwargs,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
    ):
        self.save_init_params(locals())
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.encoder=CNN(
            self.imsize,
            self.imsize,
            self.input_channels,
            *conv_args,
            **conv_kwargs)
        self.fc1 = nn.Linear(self.encoder.output_size, representation_size)
        self.fc2 = nn.Linear(self.encoder.output_size, representation_size)
        self.hidden_init(self.fc1.weight)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)
        self.decoder = DCNN(
            *deconv_args,
            input_size=representation_size,
            **deconv_kwargs)
        self.epoch = 0

    def encode(self, input):
        h = self.encoder(input)
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)
