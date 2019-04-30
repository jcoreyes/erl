import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
import numpy as np
from railrl.torch.networks import CNN, TwoHeadDCNN, DCNN
from railrl.torch.vae.vae_base import compute_bernoulli_log_prob, compute_gaussian_log_prob, GaussianLatentVAE
from railrl.torch.vae.conv_vae import ConvVAE

class ConditionalConvVAE(GaussianLatentVAE):
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
            min_variance=1e-3,
            hidden_init=ptu.fanin_init,
            reconstruction_channels=3,
            base_depth=32,
            weight_init_gain=1.0,

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
        super().__init__(representation_size)
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.input_channels = 6
        self.imsize = imsize
        self.imlength = self.imsize*self.imsize*self.input_channels
        self.hidden_init = hidden_init
        self.init_w = init_w
        self.architecture = architecture
        self.reconstruction_channels = reconstruction_channels
        self.decoder_output_activation = decoder_output_activation

        conv_args, conv_kwargs, deconv_args, deconv_kwargs = \
            architecture['conv_args'], architecture['conv_kwargs'], \
            architecture['deconv_args'], architecture['deconv_kwargs']
        conv_output_size=deconv_args['deconv_input_width']*\
                         deconv_args['deconv_input_height']*\
                         deconv_args['deconv_input_channels']

        # self.encoder=encoder_class(
        #     **conv_args,
        #     paddings=np.zeros(len(conv_args['kernel_sizes']), dtype=np.int64),
        #     input_height=self.imsize,
        #     input_width=self.imsize,
        #     input_channels=self.input_channels,
        #     output_size=conv_output_size,
        #     init_w=init_w,
        #     hidden_init=hidden_init,
        #     **conv_kwargs)

        # self.decoder = decoder_class(
        #     **deconv_args,
        #     fc_input_size=representation_size,
        #     init_w=init_w,
        #     output_activation=decoder_output_activation,
        #     paddings=np.zeros(len(deconv_args['kernel_sizes']), dtype=np.int64),
        #     hidden_init=hidden_init,
        #     **deconv_kwargs)

        self.relu = nn.LeakyReLU()
        self.gain = weight_init_gain
        self.init_w = init_w

        self.base_depth = base_depth
        self.epoch = 0
        self.decoder_distribution=decoder_distribution
        self.representation_size = representation_size

        self._create_layers()

    def _create_layers(self):
        self.conv1 = nn.Conv2d(3, self.base_depth, 3, stride=3)
        nn.init.xavier_uniform_(self.conv1.weight, gain=self.gain)
        self.conv2 = nn.Conv2d(self.base_depth, self.base_depth * 2 , 3, stride=3)
        nn.init.xavier_uniform_(self.conv2.weight, gain=self.gain)
        self.conv3 = nn.Conv2d(2 * self.base_depth* 2, self.base_depth * 4, 3, stride=2) # fusion
        nn.init.xavier_uniform_(self.conv3.weight, gain=self.gain)

        self.fc1 = nn.Linear(self.base_depth*4*2*2, self.representation_size)
        self.fc2 = nn.Linear(self.base_depth*4*2*2, self.representation_size)

        self.fc1.weight.data.uniform_(-self.init_w, self.init_w)
        self.fc1.bias.data.uniform_(-self.init_w, self.init_w)

        self.fc2.weight.data.uniform_(-self.init_w, self.init_w)
        self.fc2.bias.data.uniform_(-self.init_w, self.init_w)

        self.deconv_fc1 = nn.Linear(self.representation_size, 2*2*self.base_depth*4)
        self.deconv_fc1.weight.data.uniform_(-self.init_w, self.init_w)
        self.deconv_fc1.bias.data.uniform_(-self.init_w, self.init_w)

        self.dconv1 = nn.ConvTranspose2d(self.base_depth*4, self.base_depth*4, 5, stride=3)
        nn.init.xavier_uniform_(self.dconv1.weight, gain=self.gain)

        self.dconv2 = nn.ConvTranspose2d(2*self.base_depth*4, self.base_depth*2, 6, stride=2) # skip connection
        nn.init.xavier_uniform_(self.dconv2.weight, gain=self.gain)

        self.dconv3 = nn.ConvTranspose2d(2*self.base_depth*2, 3, 10, stride=2)
        nn.init.xavier_uniform_(self.dconv3.weight, gain=self.gain)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=4)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=4)

    def forward(self, input):
        """
        :param input:
        :return: reconstructed input, obs_distribution_params, latent_distribution_params
        """
        input = input.view(-1, self.input_channels, self.imsize, self.imsize)
        # import pdb; pdb.set_trace()
        x = input[:, :3, :, :]
        x0 = input[:, 3:, :, :]

        a1 = self.conv1(x)
        a2 = self.conv2(self.relu(a1))
        b1 = self.conv1(x0)
        b2 = self.conv2(self.relu(b1)) # 32 x 18 x 18

        h2 = torch.cat((a2, b2), dim=1)
        h3 = self.conv3(self.relu(h2))

        # hlayers = [a1, h2, h3, ]
        # for l in hlayers:
        #     print(l.shape)
        h = h3.view(h3.size()[0], -1)

        ### encode
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        latent_distribution_params = (mu, logvar)

        ### reparameterize

        latents = self.reparameterize(latent_distribution_params)

        dh0 = self.deconv_fc1(latents)
        dh0 = self.relu(dh0.view(-1, self.base_depth*4, 2, 2))
        ### decode

        dh1 = self.dconv1(dh0)
        dh1 = torch.cat((dh1, self.up2(dh0)), dim=1)

        dh2 = self.dconv2(self.relu(dh1))

        # fusion
        f = torch.cat((dh2, self.up1(b2)), dim=1)

        dh3 = self.dconv3(self.relu(f))

        # print(dh3.shape)

        """
        f3 = torch.cat((dh3, b2), dim=1)
        dh4 = self.dconv4(self.relu(f3))
        dh5 = self.dconv5(self.relu(dh4))
        """

        # dlayers = [dh1, dh2, dh3, dh4, dh5]
        # for l in dlayers:
        #     print(l.shape)

        decoded = self.decoder_output_activation(dh3)

        decoded = decoded.view(-1, self.imsize*self.imsize*self.reconstruction_channels)
        # if self.decoder_distribution == 'bernoulli': # assume bernoulli
        reconstructions, obs_distribution_params = decoded, [decoded]

        return reconstructions, obs_distribution_params, latent_distribution_params

    def encode(self, input):
        input = input.view(-1, self.input_channels, self.imsize, self.imsize)
        x = input[:, :3, :, :]
        x0 = input[:, 3:, :, :]
        a1 = self.conv1(x)
        a2 = self.conv2(self.relu(a1))
        b1 = self.conv1(x0)
        b2 = self.conv2(self.relu(b1)) # 32 x 18 x 18

        h2 = torch.cat((a2, b2), dim=1)
        h3 = self.conv3(self.relu(h2))

        # hlayers = [h2, h3, h4, h5]
        # for l in hlayers:
        #     print(l.shape)
        h = h3.view(h3.size()[0], -1)

        ### encode
        mu = self.fc1(h)
        if self.log_min_variance is None:
            logvar = self.fc2(h)
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        return (mu, logvar)


    def decode(self, latents, data):
        data = data.view(-1, self.input_channels, self.imsize, self.imsize)
        x0 = data[:, 3:, :, :]

        b1 = self.conv1(x0)
        b2 = self.conv2(self.relu(b1)) # 32 x 18 x 18

        dh0 = self.deconv_fc1(latents)
        dh0 = self.relu(dh0.view(-1, self.base_depth*4, 2, 2))
        dh1 = self.dconv1(dh0)
        dh1 = torch.cat((dh1, self.up2(dh0)), dim=1)
        dh2 = self.dconv2(self.relu(dh1))

        f = torch.cat((dh2, self.up1(b2)), dim=1)

        dh3 = self.dconv3(self.relu(f))

        #f3 = torch.cat((dh3, b2), dim=1)
        #dh4 = self.dconv4(self.relu(f3))
        #dh5 = self.dconv5(self.relu(dh4))

        #decoded = self.decoder_output_activation(dh5)
        decoded = self.decoder_output_activation(dh3)
        decoded = decoded.view(-1, self.imsize*self.imsize*self.reconstruction_channels)
        if self.decoder_distribution == 'bernoulli':
            return decoded, [decoded]
        elif self.decoder_distribution == 'gaussian_identity_variance':
            return torch.clamp(decoded, 0, 1), [torch.clamp(decoded, 0, 1), torch.ones_like(decoded)]
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))

    def logprob(self, inputs, obs_distribution_params):
        if self.decoder_distribution == 'bernoulli':
            inputs = inputs.view(
                -1, self.input_channels, self.imsize, self.imsize
            )
            length = self.reconstruction_channels * self.imsize * self.imsize
            x = inputs[:, :self.reconstruction_channels, :, :].view(-1, length)
            # x = x.narrow(start=0, length=length,
                 # dim=1).contiguous().view(-1, length)
            reconstruction_x = obs_distribution_params[0]
            log_prob = compute_bernoulli_log_prob(x, reconstruction_x) * (self.imsize*self.imsize*self.reconstruction_channels)
            return log_prob
        if self.decoder_distribution == 'gaussian_identity_variance':
            inputs = inputs.narrow(start=0, length=self.imlength // 2,
                                   dim=1).contiguous().view(-1, self.imlength // 2)
            log_prob = -1*F.mse_loss(inputs, obs_distribution_params[0],reduction='elementwise_mean')
            return log_prob
        else:
            raise NotImplementedError('Distribution {} not supported'.format(self.decoder_distribution))
