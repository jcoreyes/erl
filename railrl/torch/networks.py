"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision.models as models

from railrl.policies.base import Policy
from railrl.pythonplusplus import identity
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import PyTorchModule, eval_np
from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.modules import SelfOuterProductLinear, LayerNorm

import numpy as np


class PretrainedCNN(PyTorchModule):
    # Uses a pretrained CNN architecture from torchvision
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            torchvision_architecture=models.vgg19_bn,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels

        self.pretrained_model = nn.Sequential(*list(torchvision_architecture(pretrained=True).children())[:-1])
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        test_mat = self.pretrained_model(test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                fc_input_size = hidden_size

                fc_layer.weight.data.uniform_(-init_w, init_w)
                fc_layer.bias.data.uniform_(-init_w, init_w)

                self.fc_layers.append(fc_layer)

                if self.batch_norm_fc:
                    norm_layer = nn.BatchNorm1d(hidden_size)
                    self.fc_norm_layers.append(norm_layer)

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward_conv(h)

        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        return self.pretrained_model(h)

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.batch_norm_fc:
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h


class CNN(PyTorchModule):
    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes=None,
            added_fc_input_size=0,
            batch_norm_conv=False,
            batch_norm_fc=False,
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
    ):
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.batch_norm_conv = batch_norm_conv
        self.batch_norm_fc = batch_norm_fc
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        self.pool_type = pool_type

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                self.pool_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_sizes[i],
                        stride=pool_strides[i],
                        padding=pool_paddings[i],
                    )
                )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.batch_norm_conv:
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.pool_type != 'none':
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                fc_input_size = hidden_size

                fc_layer.weight.data.uniform_(-init_w, init_w)
                fc_layer.bias.data.uniform_(-init_w, init_w)

                self.fc_layers.append(fc_layer)

                if self.batch_norm_fc:
                    norm_layer = nn.BatchNorm1d(hidden_size)
                    self.fc_norm_layers.append(norm_layer)

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_last_activations=False):
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        h = self.apply_forward_conv(h)

        if self.output_conv_channels:
            return h

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        h = self.apply_forward_fc(h)

        if return_last_activations:
            return h
        return self.output_activation(self.last_fc(h))

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.batch_norm_conv:
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.batch_norm_fc:
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h


class MergedCNN(CNN):
    '''
    CNN that supports input directly into fully connected layers
    '''

    def __init__(self,
                 added_fc_input_size,
                 **kwargs
                 ):
        super().__init__(added_fc_input_size=added_fc_input_size,
                         **kwargs)

    def forward(self, conv_input, fc_input):
        h = torch.cat((conv_input, fc_input), dim=1)
        output = super().forward(h)
        return output


class Split(nn.Module):
    """
    Split input and process each chunk with a separate module.
    """
    def __init__(self, module1, module2, split_idx):
        super().__init__()
        self.module1 = module1
        self.module2 = module2
        self.split_idx = split_idx

    def forward(self, x):
        in1 = x[:, :self.split_idx]
        out1 = self.module1(in1)

        in2 = x[:, self.split_idx:]
        out2 = self.module2(in2)

        return out1, out2


class FlattenEach(nn.Module):
    def forward(self, inputs):
        return tuple(x.view(x.size(0), -1) for x in inputs)


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.view(inputs.size(0), -1)


class Concat(nn.Module):
    def forward(self, inputs):
        return torch.cat(inputs, dim=1)


class CNNPolicy(CNN, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class Mlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dim 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpQf(FlattenMlp):
    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            action_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer

    def forward(self, obs, actions, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        if self.action_normalizer:
            actions = self.action_normalizer.normalize(actions)
        return super().forward(obs, actions, **kwargs)


class MlpQfWithObsProcessor(Mlp):
    def __init__(self, obs_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_processor = obs_processor

    def forward(self, obs, actions, **kwargs):
        h = self.obs_processor(obs)
        flat_inputs = torch.cat((h, actions), dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


class ImageStatePolicy(PyTorchModule, Policy):
    """Switches between image or state inputs"""

    def __init__(
            self,
            image_conv_net,
            state_fc_net,
    ):
        super().__init__()

        assert image_conv_net is None or state_fc_net is None
        self.image_conv_net = image_conv_net
        self.state_fc_net = state_fc_net

    def forward(self, input, return_preactivations=False):
        if self.image_conv_net is not None:
            image = input[:, :21168]
            return self.image_conv_net(image)
        if self.state_fc_net is not None:
            state = input[:, 21168:]
            return self.state_fc_net(state)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class ImageStateQ(PyTorchModule):
    """Switches between image or state inputs"""

    def __init__(
            self,
            # obs_dim,
            # action_dim,
            # goal_dim,
            image_conv_net,  # assumed to be a MergedCNN
            state_fc_net,
    ):
        super().__init__()

        assert image_conv_net is None or state_fc_net is None
        # self.obs_dim = obs_dim
        # self.action_dim = action_dim
        # self.goal_dim = goal_dim
        self.image_conv_net = image_conv_net
        self.state_fc_net = state_fc_net

    def forward(self, input, action, return_preactivations=False):
        if self.image_conv_net is not None:
            image = input[:, :21168]
            return self.image_conv_net(image, action)
        if self.state_fc_net is not None:
            state = input[:, 21168:]  # action + state
            return self.state_fc_net(state, action)


class FeedForwardQFunction(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            observation_hidden_size,
            embedded_hidden_size,
            init_w=3e-3,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            batchnorm_obs=False,
    ):
        print("WARNING: This class will soon be deprecated.")
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.observation_hidden_size = observation_hidden_size
        self.embedded_hidden_size = embedded_hidden_size
        self.hidden_init = hidden_init
        self.obs_fc = nn.Linear(obs_dim, observation_hidden_size)
        self.embedded_fc = nn.Linear(observation_hidden_size + action_dim,
                                     embedded_hidden_size)

        self.last_fc = nn.Linear(embedded_hidden_size, 1)
        self.output_activation = output_activation

        self.init_weights(init_w)
        self.batchnorm_obs = batchnorm_obs
        if self.batchnorm_obs:
            self.bn_obs = nn.BatchNorm1d(obs_dim)

    def init_weights(self, init_w):
        self.hidden_init(self.obs_fc.weight)
        self.obs_fc.bias.data.fill_(0)
        self.hidden_init(self.embedded_fc.weight)
        self.embedded_fc.bias.data.fill_(0)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs, action):
        if self.batchnorm_obs:
            obs = self.bn_obs(obs)
        h = obs
        h = F.relu(self.obs_fc(h))
        h = torch.cat((h, action), dim=1)
        h = F.relu(self.embedded_fc(h))
        return self.output_activation(self.last_fc(h))


class FeedForwardPolicy(PyTorchModule):
    def __init__(
            self,
            obs_dim,
            action_dim,
            fc1_size,
            fc2_size,
            init_w=1e-3,
            hidden_init=ptu.fanin_init,
    ):
        print("WARNING: This class will soon be deprecated.")
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.hidden_init = hidden_init

        self.fc1 = nn.Linear(obs_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.last_fc = nn.Linear(fc2_size, action_dim)

        self.init_weights(init_w)

    def init_weights(self, init_w):
        self.hidden_init(self.fc1.weight)
        self.fc1.bias.data.fill_(0)
        self.hidden_init(self.fc2.weight)
        self.fc2.bias.data.fill_(0)

        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        return F.tanh(self.last_fc(h))

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


"""
Random Networks Below
"""


class TwoHeadMlp(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            first_head_size,
            second_head_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.first_head_size = first_head_size
        self.second_head_size = second_head_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.first_head = nn.Linear(in_size, self.first_head_size)
        self.first_head.weight.data.uniform_(-init_w, init_w)

        self.second_head = nn.Linear(in_size, self.second_head_size)
        self.second_head.weight.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.first_head(h)
        first_output = self.output_activation(preactivation)
        preactivation = self.second_head(h)
        second_output = self.output_activation(preactivation)

        return first_output, second_output


class OuterProductFF(PyTorchModule):
    """
    An interesting idea that I had where you first take the outer product of
    all inputs, flatten it, and then pass it through a linear layer. I
    haven't really tested this, but I'll leave it here to tempt myself later...
    """

    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.,
    ):
        super().__init__()

        self.sops = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            sop = SelfOuterProductLinear(in_size, next_size)
            in_size = next_size
            hidden_init(sop.fc.weight)
            sop.fc.bias.data.fill_(b_init_value)
            self.__setattr__("sop{}".format(i), sop)
            self.sops.append(sop)
        self.output_activation = output_activation
        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.fill_(b_init_value)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, sop in enumerate(self.sops):
            h = self.hidden_activation(sop(h))
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class AETanhPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(
            self,
            ae,
            env,
            history_length,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs, output_activation=torch.tanh)
        self.ae = ae
        self.history_length = history_length
        self.env = env

    def get_action(self, obs_np):
        obs = obs_np
        obs = ptu.from_numpy(obs)
        image_obs, fc_obs = self.env.split_obs(obs)
        latent_obs = self.ae.history_encoder(image_obs, self.history_length)
        if fc_obs is not None:
            latent_obs = torch.cat((latent_obs, fc_obs), dim=1)
        obs_np = ptu.get_numpy(latent_obs)[0]
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}


class FeatPointMlp(PyTorchModule):
    def __init__(
            self,
            downsample_size,
            input_channels,
            num_feat_points,
            temperature=1.0,
            init_w=1e-3,
            input_size=32,
            hidden_init=ptu.fanin_init,
            output_activation=identity,
    ):
        super().__init__()

        self.downsample_size = downsample_size
        self.temperature = temperature
        self.num_feat_points = num_feat_points
        self.hidden_init = hidden_init
        self.output_activation = output_activation
        self.input_channels = input_channels
        self.input_size = input_size

        #        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(input_channels, 48, kernel_size=5, stride=2)
        #        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(48, 48, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(48, self.num_feat_points, kernel_size=5, stride=1)

        test_mat = ptu.zeros(1, self.input_channels, self.input_size, self.input_size)
        test_mat = self.conv1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.conv3(test_mat)
        self.out_size = int(np.prod(test_mat.shape))
        self.fc1 = nn.Linear(2 * self.num_feat_points, 400)
        self.fc2 = nn.Linear(400, 300)
        self.last_fc = nn.Linear(300, self.input_channels * self.downsample_size * self.downsample_size)

        self.init_weights(init_w)
        self.i = 0

    def init_weights(self, init_w):
        self.hidden_init(self.conv1.weight)
        self.conv1.bias.data.fill_(0)
        self.hidden_init(self.conv2.weight)
        self.conv2.bias.data.fill_(0)

    def forward(self, input):
        h = self.encoder(input)
        out = self.decoder(h)
        return out

    def encoder(self, input):
        x = input.contiguous().view(-1, self.input_channels, self.input_size, self.input_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
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
        #        h = x.view(-1, 2, self.num_feat_points).transpose(1, 2).contiguous().view(-1, self.num_feat_points * 2)
        h = x.view(-1, self.num_feat_points * 2)
        return h

    def decoder(self, input):
        h = input
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.last_fc(h)
        return h

    def history_encoder(self, input, history_length):
        input = input.contiguous().view(-1,
                                        self.input_channels,
                                        self.input_size,
                                        self.input_size)
        latent = self.encoder(input)

        assert latent.shape[0] % history_length == 0
        n_samples = latent.shape[0] // history_length
        latent = latent.view(n_samples, -1)
        return latent


class TwoHeadDCNN(PyTorchModule):
    def __init__(self,
                 fc_input_size,
                 hidden_sizes,

                 deconv_input_width,
                 deconv_input_height,
                 deconv_input_channels,

                 deconv_output_kernel_size,
                 deconv_output_strides,
                 deconv_output_channels,

                 kernel_sizes,
                 n_channels,
                 strides,
                 paddings,

                 batch_norm_deconv=False,
                 batch_norm_fc=False,
                 init_w=1e-3,
                 hidden_init=nn.init.xavier_uniform_,
                 hidden_activation=nn.ReLU(),
                 output_activation=identity
                 ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * self.deconv_input_height * self.deconv_input_width
        self.batch_norm_deconv = batch_norm_deconv
        self.batch_norm_fc = batch_norm_fc

        self.deconv_layers = nn.ModuleList()
        self.deconv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            norm_layer = nn.BatchNorm1d(hidden_size)
            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            self.fc_norm_layers.append(norm_layer)
            fc_input_size = hidden_size

        self.last_fc = nn.Linear(fc_input_size, deconv_input_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

        for out_channels, kernel_size, stride, padding in \
                zip(n_channels, kernel_sizes, strides, paddings):
            deconv = nn.ConvTranspose2d(deconv_input_channels,
                                        out_channels,
                                        kernel_size,
                                        stride=stride,
                                        padding=padding)
            hidden_init(deconv.weight)
            deconv.bias.data.fill_(0)

            deconv_layer = deconv
            self.deconv_layers.append(deconv_layer)
            deconv_input_channels = out_channels

        test_mat = torch.zeros(1, self.deconv_input_channels, self.deconv_input_width,
                               self.deconv_input_height)  # initially the model is on CPU (caller should then move it to GPU if
        for deconv_layer in self.deconv_layers:
            test_mat = deconv_layer(test_mat)
            self.deconv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))

        self.first_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.first_deconv_output.weight)
        self.first_deconv_output.bias.data.fill_(0)

        self.second_deconv_output = nn.ConvTranspose2d(
            deconv_input_channels,
            deconv_output_channels,
            deconv_output_kernel_size,
            stride=deconv_output_strides,
        )
        hidden_init(self.second_deconv_output.weight)
        self.second_deconv_output.bias.data.fill_(0)

    def forward(self, input):
        h = self.apply_forward(input, self.fc_layers, self.fc_norm_layers, use_batch_norm=self.batch_norm_fc)
        h = self.hidden_activation(self.last_fc(h))
        h = h.view(-1, self.deconv_input_channels, self.deconv_input_width, self.deconv_input_height)
        h = self.apply_forward(h, self.deconv_layers, self.deconv_norm_layers, use_batch_norm=self.batch_norm_deconv)
        first_output = self.output_activation(self.first_deconv_output(h))
        second_output = self.output_activation(self.second_deconv_output(h))
        return first_output, second_output

    def apply_forward(self, input, hidden_layers, norm_layers, use_batch_norm=False):
        h = input
        for layer, norm_layer in zip(hidden_layers, norm_layers):
            h = layer(h)
            if use_batch_norm:
                h = norm_layer(h)
            h = self.hidden_activation(h)
        return h


class DCNN(TwoHeadDCNN):
    def forward(self, input):
        return super().forward(input)[0]
