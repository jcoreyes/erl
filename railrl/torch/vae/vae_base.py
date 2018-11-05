import torch

from railrl.pythonplusplus import identity
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
        :return: latent_distribution_params
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def rsample(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return: latents
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def reparameterize(self, latent_distribution_params):
        """

        :param latent_distribution_params:
        :return:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def decode(self, latents):
        """
        :param latents:
        :return: reconstruction, obs_distribution_params
        """
        raise NotImplementedError()

    def logprob(self, inputs, obs_distribution_params):
        """
        :param inputs:
        :param obs_distribution_params:
        :return: log probability of input under decoder
        """
        raise NotImplementedError()

    def kl_divergence(self, latent_distribution_params):
        """
        :param latent_distribution_params:
        :return:
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

    def rsample(self, latent_distribution_params):
        mu, logvar = latent_distribution_params
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def reparameterize(self, latent_distribution_params):
        if self.training:
            return self.rsample(latent_distribution_params)
        else:
            return latent_distribution_params[0]

    def forward(self, input):
        """

        :param input:
        :return: reconstructed input, latent_distribution_params
        """
        mu, logvar = self.encode(input)
        z = self.reparameterize((mu, logvar))
        reconstructions, obs_distribution_params = self.decode(z)
        return reconstructions, obs_distribution_params, (mu, logvar)

    def kl_divergence(self, distribution_params):
        mu, logvar = distribution_params
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
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            num_latents_to_sample=1,
            hidden_init=ptu.fanin_init,
    ):
        """

        :param representation_size:
        :param conv_args:
        must be a dictionary specifying the following:
            kernel_sizes
            n_channels
            strides
        :param conv_kwargs:
        a dictionary specifying the following:
            hidden_sizes
            batch_norm
        :param deconv_args:
        must be a dictionary specifying the following:
            hidden_sizes
            deconv_input_width
            deconv_input_height
            deconv_input_channels
            deconv_output_kernel_size
            deconv_output_strides
            deconv_output_channels
            kernel_sizes
            n_channels
            strides
        :param deconv_kwargs:
            batch_norm
        :param encoder_class:
        :param decoder_class:
        :param decoder_output_activation:
        :param decoder_distribution:
        :param input_channels:
        :param imsize:
        :param init_w:
        :param min_variance:
        :param num_latents_to_sample:
        :param hidden_init:
        """
        self.save_init_params(locals())
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels

        conv_output_size=deconv_args['deconv_input_width']*\
                         deconv_args['deconv_input_height']*\
                         deconv_args['deconv_input_channels']

        self.encoder=encoder_class(
            **conv_args,
            paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
            input_height=self.imsize,
            input_width=self.imsize,
            input_channels=self.input_channels,
            output_size=conv_output_size,
            init_w=init_w,
            **conv_kwargs)

        self.fc1 = nn.Linear(self.encoder.output_size, representation_size)
        self.fc2 = nn.Linear(self.encoder.output_size, representation_size)

        hidden_init(self.fc1.weight)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        hidden_init(self.fc2.weight)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
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
        return (mu, logvar)

    def decode(self, latents):
        decoded = self.decoder(latents).view(-1, self.imsize*self.imsize*self.input_channels)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def get_sampled_latents_and_latent_distributions(self, input):
        mu, logvar = self.encode(input)
        mu = mu.view((mu.size()[0], 1, mu.size()[1]))
        stds = (0.5 * logvar).exp()
        stds = stds.view(stds.size()[0], 1, stds.size()[1])
        epsilon = ptu.randn((mu.size()[0], self.num_latents_to_sample, mu.size()[1]))
        latents = epsilon * stds + mu
        return latents, mu, logvar, stds

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'bernoulli':
            log_prob = compute_bernoulli_log_prob(inputs, obs_distribution_params[0], vector_dimension=self.imlength)
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
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-3,
            num_latents_to_sample=1,
            hidden_init=ptu.fanin_init,
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
            decoder_output_activation=decoder_output_activation,
            decoder_distribution=decoder_distribution,

            input_channels=input_channels,
            imsize=imsize,
            init_w=init_w,
            min_variance=min_variance,
            num_latents_to_sample=num_latents_to_sample,
        )

    def decode(self, latents):
        first_output, second_output = self.decoder(latents)
        first_output = first_output.view(-1, self.imsize*self.imsize*self.input_channels)
        second_output = second_output.view(-1, self.imsize*self.imsize*self.input_channels)
        if self.decoder_distribution == 'gaussian':
            return first_output, (first_output, second_output)
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'gaussian':
            latents, mu, logvar, stds = self.get_sampled_latents_and_latent_distributions(inputs)
            dec_mu, dec_var = self.model.decode(latents)[1]
            log_prob = compute_gaussian_log_prob(inputs, dec_mu, dec_var, vector_dimension=self.imlength)
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

def compute_bernoulli_log_prob(x, recon_x, vector_dimension):
    # Multiply back in the vector_dimension so the cross entropy is only averaged over the batch size
    return -1* F.binary_cross_entropy(
        recon_x,
        x.narrow(start=0, length=vector_dimension,
                 dim=1).contiguous().view(-1, vector_dimension),
        reduction='elementwise_mean',
    ) * vector_dimension

def compute_gaussian_log_prob(input, dec_mu, dec_var, vector_dimension):
    dec_mu = dec_mu.view(-1, vector_dimension)
    dec_var = dec_var.view(-1, vector_dimension)
    decoder_dist = Normal(dec_mu, dec_var.pow(0.5))
    input = input.view(-1, vector_dimension)
    log_probs = decoder_dist.log_prob(input)
    vals = log_probs.sum(dim=1, keepdim=True)
    return vals.mean()