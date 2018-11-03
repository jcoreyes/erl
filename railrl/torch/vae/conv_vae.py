# Adapted from pytorch examples

from __future__ import print_function
import torch
import torch.utils.data
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
import numpy as np
from railrl.torch.core import PyTorchModule


class ConvVAESmall(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=48,
            hidden_init=ptu.fanin_init,
            encoder_activation=identity,
            decoder_activation=identity,
            min_variance=1e-3,
            state_size=0,
            num_latents_to_sample=1,
    ):
        self.save_init_params(locals())
        super().__init__()
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.num_latents_to_sample=num_latents_to_sample
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.relu = nn.ReLU()
        self.init_w = init_w
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.kernel_out = 64
        self.conv_output_dim = self.kernel_out * 9
        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                        stride=3)
        self.epoch = 0
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        # self.fc4.weight.data.uniform_(-init_w, init_w)
        # self.fc4.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength)
        conv_input = input
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h = x.view(-1, self.conv_output_dim)  # flatten
        # h = self.relu(self.fc4(h))
        mu = self.encoder_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.encoder_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, self.kernel_out, 3, 3)
        x = F.relu(self.conv4(h))
        x = F.relu(self.conv5(x))
        x = self.conv6(x).view(-1,
                               self.imsize * self.imsize * self.input_channels)

        return self.decoder_activation(x)

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

class ConvVAESmallDouble(PyTorchModule):
    def __init__(
            self,
            representation_size,
            init_w=1e-3,
            input_channels=1,
            imsize=48,
            hidden_init=ptu.fanin_init,
            encoder_activation=identity,
            decoder_activation=identity,
            min_variance=1e-3,
            state_size=0,
            unit_variance=False,
            is_auto_encoder=False,
            variance_scaling=1,
            num_latents_to_sample=1,
    ):
        self.save_init_params(locals())
        super().__init__()
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))

        self.representation_size = representation_size
        self.variance_scaling=variance_scaling
        self.hidden_init = hidden_init
        self.encoder_activation = encoder_activation
        self.decoder_activation = decoder_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        self.num_latents_to_sample = num_latents_to_sample
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
        self.unit_variance = unit_variance
        self.relu = nn.ReLU()
        self.init_w = init_w
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.kernel_out = 64
        self.conv_output_dim = self.kernel_out * 9
        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                        stride=3)
        self.conv7 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                        stride=3)
        self.epoch = 0
        self.is_auto_encoder=is_auto_encoder
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)
        self.hidden_init(self.conv7.weight)
        self.conv7.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

        # self.fc4.weight.data.uniform_(-init_w, init_w)
        # self.fc4.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength)
        conv_input = input
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h = x.view(-1, self.conv_output_dim)  # flatten
        # h = self.relu(self.fc4(h))
        mu = self.encoder_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.encoder_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        return self.decode_mean_and_logvar(z)[0]

    def decode_mean_and_logvar(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, self.kernel_out, 3, 3)
        x = F.relu(self.conv4(h))
        h = F.relu(self.conv5(x))
        mu = self.conv6(h).view(-1,
                                self.imsize * self.imsize * self.input_channels)
        logvar = self.conv7(h).view(-1,
                                    self.imsize * self.imsize * self.input_channels)
        if self.unit_variance:
            logvar = ptu.zeros_like(logvar)
        return self.decoder_activation(mu), mu, logvar

    def forward(self, x):
        mu, logvar = self.encode(x)
        if self.is_auto_encoder:
            z = mu
        else:
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
            state_size=0,
            action_dim=None,
    ):
        self.save_init_params(locals())
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)
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
        self.conv_output_dim = 128  # kernel 3

        # self.hidden = nn.Linear(self.conv_output_dim + added_fc_size, representation_size)

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)

        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)
        self.conv4 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=6, stride=3)
        self.conv6 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                        stride=3)

        if action_dim is not None:
            self.linear_constraint_fc = \
                nn.Linear(
                    self.representation_size + action_dim,
                    self.representation_size
                )
        else:
            self.linear_constraint_fc = None

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dim=1)

        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        h = x.view(-1, 128)  # flatten
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dim=1)
            h = torch.cat((h, fc_input), dim=1)
        mu = self.output_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.output_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
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


class ConvVAELarge(PyTorchModule):
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
    ):
        self.save_init_params(locals())
        # TODO(mdalal2020): You probably want to fix this init call...
        super().__init__()
        self.representation_size = representation_size
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.imsize = imsize
        self.imlength = self.imsize ** 2 * self.input_channels
        if min_variance is None:
            self.log_min_variance = None
        else:
            self.log_min_variance = float(np.log(min_variance))
        self.dist_mu = np.zeros(self.representation_size)
        self.dist_std = np.ones(self.representation_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.added_fc_size = added_fc_size
        self.init_w = init_w

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, stride=3)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn6 = nn.BatchNorm2d(32)

        self.conv_output_dim = 128

        self.fc1 = nn.Linear(self.conv_output_dim, representation_size)
        self.fc2 = nn.Linear(self.conv_output_dim, representation_size)

        self.fc3 = nn.Linear(representation_size, self.conv_output_dim)

        self.conv7 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2)
        self.conv8 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv9 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=3)
        self.conv10 = nn.ConvTranspose2d(32, 32, kernel_size=5, stride=1)
        self.conv11 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=1)
        self.conv12 = nn.ConvTranspose2d(16, input_channels, kernel_size=6,
                                         stride=1)
        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)
        self.hidden_init(self.conv3.weight)
        self.conv3.bias.data.fill_(0)
        self.hidden_init(self.conv4.weight)
        self.conv4.bias.data.fill_(0)
        self.hidden_init(self.conv5.weight)
        self.conv5.bias.data.fill_(0)
        self.hidden_init(self.conv6.weight)
        self.conv6.bias.data.fill_(0)

        self.hidden_init(self.conv7.weight)
        self.conv7.bias.data.fill_(0)
        self.hidden_init(self.conv8.weight)
        self.conv8.bias.data.fill_(0)
        self.hidden_init(self.conv9.weight)
        self.conv9.bias.data.fill_(0)
        self.hidden_init(self.conv10.weight)
        self.conv10.bias.data.fill_(0)
        self.hidden_init(self.conv11.weight)
        self.conv11.bias.data.fill_(0)
        self.hidden_init(self.conv12.weight)
        self.conv12.bias.data.fill_(0)

        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.fc1.weight.data.uniform_(-init_w, init_w)
        self.fc1.bias.data.uniform_(-init_w, init_w)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-init_w, init_w)
        self.fc2.bias.data.uniform_(-init_w, init_w)

        self.hidden_init(self.fc3.weight)
        self.fc3.bias.data.fill_(0)
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def encode(self, input):
        input = input.view(-1, self.imlength + self.added_fc_size)
        conv_input = input.narrow(start=0, length=self.imlength, dim=1)

        # batch_size = input.size(0)
        x = conv_input.contiguous().view(-1, self.input_channels, self.imsize,
                                         self.imsize)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        h = x.view(-1, 128)  # flatten
        if self.added_fc_size != 0:
            fc_input = input.narrow(start=self.imlength, length=self.added_fc_size, dim=1)
            h = torch.cat((h, fc_input), dim=1)
        mu = self.output_activation(self.fc1(h))
        if self.log_min_variance is None:
            logvar = self.output_activation(self.fc2(h))
        else:
            logvar = self.log_min_variance + torch.abs(self.fc2(h))
        return mu, logvar

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        h = h3.view(-1, 32, 2, 2)
        x = F.relu(self.conv7(h))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = self.conv12(x).view(-1,
                                self.imsize * self.imsize * self.input_channels)
        return self.sigmoid(x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

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

class AutoEncoder(ConvVAE):
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu
        return self.decode(z), mu, logvar


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


if __name__ == "__main__":
    m = ConvVAE(2)
    for epoch in range(10):
        m.train_epoch(epoch)
        m.test_epoch(epoch)
        m.dump_samples(epoch)
