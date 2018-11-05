import torch
from railrl.torch.core import PyTorchModule
import numpy as np
import abc
from railrl.torch.networks import CNN, DCNN, TwoHeadDCNN
from torch import nn
import railrl.torch.pytorch_util as ptu
from torch.distributions import Normal
from torch.nn import functional as F

class VAEBase(PyTorchModule,  metaclass=abc.ABCMeta):
    def __init__(
            self,
            representation_size,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size

    @abc.abstractmethod
    def encode(self, input):
        """
        :param input:
        :return: mu, logvar
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, latents):
        """
        :param latents:
        :return: reconstruction
        """
        raise NotImplementedError()

    def logprob(self, input):
        """
        :param input:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

class GaussianLatentVAE(VAEBase):
    def __init__(
            self,
            representation_size,
    ):
        self.save_init_params(locals())
        super().__init__(representation_size)
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

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

class ConvVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            conv_args,
            conv_kwargs,
            deconv_args,
            deconv_kwargs,
            encoder_class=CNN,
            decoder_class=DCNN,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            num_latents_to_sample=1,
            decoder_distribution='bernoulli',
    ):
        self.save_init_params(locals())
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels
        self.encoder=encoder_class(
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

        self.decoder = decoder_class(
            *deconv_args,
            fc_input_size=representation_size,
            **deconv_kwargs)
        self.epoch = 0
        self.num_latents_to_sample = num_latents_to_sample
        self.decoder_distribution=decoder_distribution

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

    def compute_bernoulli_log_prob(self, recon_x, x):
        # Divide by batch_size rather than setting size_average=True because
        # otherwise the averaging will also happen across dim 1 (the
        # pixels)
        return -1* F.binary_cross_entropy(
            recon_x,
            x.narrow(start=0, length=self.imlength,
                     dim=1).contiguous().view(-1, self.imlength),
            size_average=False,
        ) / self.batch_size

    def logprob(self, input):
        """
        :param input:
        :return: log probability of input under decoder
        """
        if self.decoder_distribution == 'bernoulli':
            recon_batch, _, _ = self(input)
            log_prob = self.compute_bernoulli_log_prob(recon_batch, input)
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

class ConvVAEDouble(ConvVAE):
    def __init__(
            self,
            representation_size,
            conv_args,
            conv_kwargs,
            deconv_args,
            deconv_kwargs,
            encoder_class=CNN,
            decoder_class=TwoHeadDCNN,
            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            num_latents_to_sample=1,
    ):
        self.save_init_params(locals())
        super().__init__(
            representation_size,
            conv_args,
            conv_kwargs,
            deconv_args,
            deconv_kwargs,
            encoder_class=encoder_class,
            decoder_class=decoder_class,
            input_channels=input_channels,
            imsize=imsize,
            init_w=init_w,
            min_variance=min_variance,
            num_latents_to_sample=num_latents_to_sample,
        )

    def decode(self, latents):
        return self.decode_all_outputs(latents)[0]

    def decode_all_outputs(self, latents):
        first_output, second_output = self.decoder(latents)
        first_output = first_output.view(-1, self.imsize*self.imsize*self.input_channels)
        second_output = second_output.view(-1, self.imsize*self.imsize*self.input_channels)
        return first_output, second_output

    def compute_gaussian_log_prob(self, input, dec_mu, dec_var):
        dec_mu = dec_mu.view(-1, self.input_channels*self.imsize**2)
        dec_var = dec_var.view(-1, self.input_channels*self.imsize**2)
        decoder_dist = Normal(dec_mu, dec_var.pow(0.5))
        input = input.view(-1, self.input_channels*self.imsize**2)
        log_probs = decoder_dist.log_prob(input)
        vals = log_probs.sum(dim=1, keepdim=True)
        return vals.mean()

    def logprob(self, input):
        """
        :param input:
        :return: log probability of input under decoder
        """
        if self.decoder_distribution == 'gaussian':
            latents, mu, logvar, stds = self.get_sampled_latents_and_latent_distributions(input)
            dec_mu, dec_var = self.model.decode_all_outputs(latents)
            log_prob = self.compute_gaussian_log_prob(input, dec_mu, dec_var)
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))
