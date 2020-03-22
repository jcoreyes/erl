"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from railrl.torch.networks.basic import (
    Clamp, Concat, Flatten, FlattenEach, Split,
)
from railrl.torch.networks.cnn import CNN, MergedCNN, CNNPolicy
from railrl.torch.networks.dcnn import DCNN, TwoHeadDCNN
from railrl.torch.networks.deprecated_feedforward import (
    FeedForwardPolicy, FeedForwardQFunction
)
from railrl.torch.networks.feat_point_mlp import FeatPointMlp
from railrl.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from railrl.torch.networks.linear_transform import LinearTransform
from railrl.torch.networks.mlp import (
    Mlp, FlattenMlp, MlpPolicy, TanhMlpPolicy,
    MlpQf,
    MlpQfWithObsProcessor,
)
from railrl.torch.networks.pretrained_cnn import PretrainedCNN
from railrl.torch.networks.two_headed_mlp import TwoHeadMlp

__all__ = [
    'Clamp',
    'Concat',
    'CNN',
    'CNNPolicy',
    'DCNN',
    'FeedForwardPolicy',
    'FeedForwardQFunction',
    'FeatPointMlp',
    'Flatten',
    'FlattenEach',
    'FlattenMlp',
    'LinearTransform',
    'ImageStatePolicy',
    'ImageStateQ',
    'MergedCNN',
    'Mlp',
    'PretrainedCNN',
    'Split',
    'TwoHeadDCNN',
    'TwoHeadMlp',
]

