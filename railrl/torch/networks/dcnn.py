import torch
from torch import nn as nn

from railrl.pythonplusplus import identity
from railrl.torch.core import PyTorchModule


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

                 deconv_normalization_type='none',
                 fc_normalization_type='none',
                 init_w=1e-3,
                 hidden_init=nn.init.xavier_uniform_,
                 hidden_activation=nn.ReLU(),
                 output_activation=identity
                 ):
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert deconv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation

        self.deconv_input_width = deconv_input_width
        self.deconv_input_height = deconv_input_height
        self.deconv_input_channels = deconv_input_channels
        deconv_input_size = self.deconv_input_channels * self.deconv_input_height * self.deconv_input_width
        self.deconv_normalization_type = deconv_normalization_type
        self.fc_normalization_type = fc_normalization_type

        self.deconv_layers = nn.ModuleList()
        self.deconv_norm_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for idx, hidden_size in enumerate(hidden_sizes):
            fc_layer = nn.Linear(fc_input_size, hidden_size)

            fc_layer.weight.data.uniform_(-init_w, init_w)
            fc_layer.bias.data.uniform_(-init_w, init_w)

            self.fc_layers.append(fc_layer)
            if self.fc_normalization_type == 'batch':
                self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
            if self.fc_normalization_type == 'layer':
                self.fc_norm_layers.append(nn.LayerNorm(hidden_size))
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
            if self.deconv_normalization_type == 'batch':
                self.deconv_norm_layers.append(
                    nn.BatchNorm2d(test_mat.shape[1])
                )
            if self.deconv_normalization_type == 'layer':
                self.deconv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))

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
        h = self.apply_forward(input, self.fc_layers, self.fc_norm_layers,
                               normalization_type=self.fc_normalization_type)
        h = self.hidden_activation(self.last_fc(h))
        h = h.view(-1, self.deconv_input_channels, self.deconv_input_width, self.deconv_input_height)
        h = self.apply_forward(h, self.deconv_layers,
                               self.deconv_norm_layers,
                               normalization_type=self.deconv_normalization_type)
        first_output = self.output_activation(self.first_deconv_output(h))
        second_output = self.output_activation(self.second_deconv_output(h))
        return first_output, second_output

    def apply_forward(self, input, hidden_layers, norm_layers,
                      normalization_type='none'):
        h = input
        for i, layer in enumerate(hidden_layers):
            h = layer(h)
            if normalization_type != 'none':
                h = norm_layers[i](h)
            h = self.hidden_activation(h)
        return h


class DCNN(TwoHeadDCNN):
    def forward(self, input):
        return super().forward(input)[0]
