import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
import numpy as np
from railrl.torch.networks import CNN, TwoHeadDCNN, DCNN
from railrl.torch.vae.vae_base import compute_bernoulli_log_prob, compute_gaussian_log_prob, GaussianLatentVAE, \
    compute_beta_log_prob

###### DEFAULT ARCHITECTURES #########

imsize48_default_architecture=dict(
        conv_args = dict(
            kernel_sizes=[5, 3, 3],
            n_channels=[16, 32, 64],
            strides=[3, 2, 2],
        ),
        conv_kwargs=dict(
            hidden_sizes=[],
            batch_norm_conv=False,
            batch_norm_fc=False,
        ),
        deconv_args=dict(
            hidden_sizes=[],

            deconv_input_width=3,
            deconv_input_height=3,
            deconv_input_channels=64,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[3,3],
            n_channels=[32, 16],
            strides=[2,2],
        ),
        deconv_kwargs=dict(
            batch_norm_deconv=False,
            batch_norm_fc=False,
        )
    )

imsize48_default_architecture_with_more_hidden_layers = dict(
        conv_args=dict(
            kernel_sizes=[5, 3, 3],
            n_channels=[16, 32, 64],
            strides=[3, 2, 2],
        ),
        conv_kwargs=dict(
            hidden_sizes=[500, 300, 150],
        ),
        deconv_args=dict(
            hidden_sizes=[150, 300, 500],

            deconv_input_width=3,
            deconv_input_height=3,
            deconv_input_channels=64,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[3, 3],
            n_channels=[32, 16],
            strides=[2, 2],
        ),
        deconv_kwargs=dict(
        )
    )

imsize84_default_architecture=dict(
        conv_args = dict(
            kernel_sizes=[5, 5, 5],
            n_channels=[16, 32, 32],
            strides=[3, 3, 3],
        ),
        conv_kwargs=dict(
            hidden_sizes=[],
            batch_norm_conv=True,
            batch_norm_fc=False,
        ),
        deconv_args=dict(
            hidden_sizes=[],

            deconv_input_width=2,
            deconv_input_height=2,
            deconv_input_channels=32,

            deconv_output_kernel_size=6,
            deconv_output_strides=3,
            deconv_output_channels=3,

            kernel_sizes=[5,6],
            n_channels=[32, 16],
            strides=[3,3],
        ),
        deconv_kwargs=dict(
            batch_norm_deconv=False,
            batch_norm_fc=False,
        )
    )


class ConvVAE(GaussianLatentVAE):
    def __init__(
            self,
            representation_size,
            architecture,

            encoder_class=CNN,
            decoder_class=DCNN,
            decoder_output_activation=identity,
            decoder_distribution='bernoulli',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-4,
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

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
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
            hidden_init=hidden_init,
            **conv_kwargs)

        self.fc1 = nn.Linear(self.encoder.output_size, representation_size)
        self.fc2 = nn.Linear(self.encoder.output_size, representation_size)

        hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)

        hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.decoder = decoder_class(
            **deconv_args,
            fc_input_size=representation_size,
            init_w=init_w,
            output_activation=decoder_output_activation,
            paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
            hidden_init=hidden_init,
            **deconv_kwargs)

        self.epoch = 0
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
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return decoded, [decoded, torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.narrow(start=0, length=self.imlength,
                 dim=1).contiguous().view(-1, self.imlength)
            log_prob = compute_bernoulli_log_prob(inputs, obs_distribution_params[0]) * self.imlength
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength,
                                   dim=1).contiguous().view(-1, self.imlength)
            log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0],reduction='elementwise_mean')
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

class ConvVAEDouble(ConvVAE):
    def __init__(
            self,
            representation_size,
            architecture,

            encoder_class=CNN,
            decoder_class=TwoHeadDCNN,
            decoder_output_activation=identity,
            decoder_distribution='gaussian',

            input_channels=1,
            imsize=48,
            init_w=1e-3,
            min_variance=1e-4,
            hidden_init=ptu.fanin_init,
            min_log_clamp=0,
    ):
        self.save_init_params(locals())
        super().__init__(
            representation_size,
            architecture,

            encoder_class=encoder_class,
            decoder_class=decoder_class,
            decoder_output_activation=decoder_output_activation,
            decoder_distribution=decoder_distribution,

            input_channels=input_channels,
            imsize=imsize,
            init_w=init_w,
            min_variance=min_variance,
            hidden_init=hidden_init,
        )
        self.min_log_var = min_log_clamp

    def decode(self, latents):
        first_output, second_output = self.decoder(latents)
        first_output = first_output.view(-1, self.imsize*self.imsize*self.input_channels)
        second_output = second_output.view(-1, self.imsize*self.imsize*self.input_channels)
        if self.decoder_distribution == 'gaussian':
            second_output = torch.clamp(second_output, self.min_log_var, 1)
            return first_output, (first_output, second_output)
        elif self.decoder_distribution == 'beta':
            alpha = first_output.exp()
            beta = second_output.exp()
            reconstructions = alpha/(alpha+beta)
            return reconstructions, (first_output, second_output)
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'gaussian':
            dec_mu, dec_logvar = obs_distribution_params
            dec_mu = dec_mu.view(-1, self.imlength)
            dec_var = dec_logvar.view(-1, self.imlength).exp()
            inputs = inputs.view(-1, self.imlength)
            log_prob = compute_gaussian_log_prob(inputs, dec_mu, dec_var)
            return log_prob
        elif self.decoder_distribution == 'beta':
            log_alpha, log_beta = obs_distribution_params
            alpha = log_alpha.view(-1, self.imlength).exp()
            beta = log_beta.view(-1, self.imlength).exp()
            inputs = inputs.view(-1, self.imlength)
            log_prob = compute_beta_log_prob(inputs, alpha, beta)
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

class AutoEncoder(ConvVAE):
    def forward(self, x):
        mu, logvar = self.encode(input)
        reconstructions, obs_distribution_params = self.decode(mu)
        return reconstructions, obs_distribution_params, (mu, logvar)


class SpatialAutoEncoder(ConvVAE):
    def __init__(
            self,
            representation_size,
            num_feat_points,
            *args,
            temperature=1.0,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(representation_size, *args, **kwargs)
        self.num_feat_points = num_feat_points
        self.conv3 = nn.Conv2d(32, self.num_feat_points, kernel_size=5,
                               stride=3)
        #        self.bn3 = nn.BatchNorm2d(32)

        test_mat = torch.zeros(1, self.input_channels, self.imsize, self.imsize)
        test_mat = self.conv1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.conv3(test_mat)
        self.out_size = int(np.prod(test_mat.shape))

        self.spatial_fc = nn.Linear(
            2 * self.num_feat_points + self.added_fc_size, 64)

        # self.conv_output_dim = 1568 # kernel 2
        self.conv_output_dim = 128  # kernel 3

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
        conv_input = input.narrow(start=0, length=self.imlength, dim=1)

        # batch_size = input.size(0)
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        d = int((self.out_size // self.num_feat_points) ** (1 / 2))
        x = x.view(-1, self.num_feat_points, d * d)
        x = F.softmax(x / self.temperature, 2)
        x = x.view(-1, self.num_feat_points, d, d)

        maps_x = torch.sum(x, 2)
        maps_y = torch.sum(x, 3)

        weights = ptu.from_numpy(np.arange(d) / (d + 1))

        fp_x = torch.sum(maps_x * weights, 2)
        fp_y = torch.sum(maps_y * weights, 2)

        x = torch.cat([fp_x, fp_y], 1)
        h = x.view(-1, self.num_feat_points * 2)
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dim=1)
            h = torch.cat((h, fc_input), dim=1)
        h = F.relu(self.spatial_fc(h))
        mu = self.output_activation(self.fc1(h))
        logvar = self.output_activation(self.fc2(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        return mu
