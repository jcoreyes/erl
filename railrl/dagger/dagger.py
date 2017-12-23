from collections import OrderedDict

import numpy as np
from torch import optim as optim

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch import pytorch_util as ptu
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm


class Dagger(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            model,
            mpc_controller,
            learning_rate=1e-3,
            num_paths_random=10,  # for normalizing input
            **kwargs
    ):
        super().__init__(
            env,
            mpc_controller,
            **kwargs
        )
        assert self.collection_mode == 'batch'
        self.model = model
        self.learning_rate = learning_rate
        self.num_paths_random = num_paths_random
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def _do_training(self):
        # TODO: loop through entire training set each training call
        # TODO: normalize input at beginning
        batch = self.get_batch()
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        next_obs_delta_pred = self.model(obs, actions)
        next_obs_delta = next_obs - obs
        errors = (next_obs_delta_pred - next_obs_delta) ** 2
        loss = errors.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics['Model Loss'] = np.mean(ptu.get_numpy(loss))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Obs Deltas',
                ptu.get_numpy(next_obs_delta),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Predicted Obs Deltas',
                ptu.get_numpy(next_obs_delta_pred),
            ))

    @property
    def networks(self):
        return [
            self.model
        ]
