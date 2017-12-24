from railrl.torch.data_management.normalizer import TorchFixedNormalizer
from railrl.torch.networks import FlattenMlp


class DynamicsModel(FlattenMlp):
    def __init__(
            self,
            observation_dim,
            action_dim,
            obs_normalizer: TorchFixedNormalizer=None,
            action_normalizer: TorchFixedNormalizer=None,
            **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(
            input_size=observation_dim + action_dim,
            output_size=observation_dim,
            **kwargs
        )
        self.obs_normalizer = obs_normalizer
        self.action_normalizer = action_normalizer

    def forward(self, observations, actions):
        if self.obs_normalizer is not None:
            observations = self.obs_normalizer.normalize(observations)
        if self.action_normalizer is not None:
            actions = self.action_normalizer.normalize(actions)
        obs_delta_predicted = super().forward(observations, actions)
        return self.obs_normalizer.denormalize_scale(obs_delta_predicted)


