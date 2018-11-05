import torch
from railrl.torch.core import PyTorchModule
import numpy as np
import abc
from railrl.torch.networks import CNN, DCNN
from torch import nn
import railrl.torch.pytorch_util as ptu


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
    def decode(self, latents):
        """
        :param latents:
        :return: reconstruction
        """
        raise NotImplementedError()

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def logprob(self, input):
        """
        :param input:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

    def kl_divergence(self, mu, logvar):
        return - torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

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
            num_latents_to_sample=1,
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
            fc_input_size=representation_size,
            **deconv_kwargs)
        self.epoch = 0
        self.num_latents_to_sample = num_latents_to_sample

    def encode(self, input):
        h = self.encoder(input)
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def decode(self, latents):
        return self.decoder(latents).view(-1, self.imsize*self.imsize*self.input_channels)

    def get_sampled_latents_and_latent_distributions(self, input):
        mu, logvar = self.encode(input)
        mu = mu.view((mu.size()[0], 1, mu.size()[1]))
        stds = (0.5 * logvar).exp()
        stds = stds.view(stds.size()[0], 1, stds.size()[1])
        epsilon = ptu.randn((mu.size()[0], self.num_latents_to_sample, mu.size()[1]))
        latents = epsilon * stds + mu
        return latents, mu, logvar, stds
