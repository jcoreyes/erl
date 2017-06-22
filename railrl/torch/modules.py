"""
Contain some self-contained modules. Maybe depend on pytorch_util.
"""
import torch.nn as nn
from railrl.torch import pytorch_util as ptu


class OuterProductLinear(nn.Module):
    def __init__(self, in_features1, in_features2, out_features, bias=True):
        super().__init__()
        self.fc = nn.Linear(
            (in_features1 + 1) * (in_features2 + 1),
            out_features,
            bias=bias,
        )

    def forward(self, in1, in2):
        out_product_flat = ptu.double_moments(in1, in2)
        return self.fc(out_product_flat)


class SelfOuterProductLinear(OuterProductLinear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, in_features, out_features, bias=bias)

    def forward(self, input):
        return super().forward(input, input)
