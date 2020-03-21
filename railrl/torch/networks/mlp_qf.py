import torch

from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.networks import Mlp, FlattenMlp


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


