from collections import OrderedDict
import numpy as np

from torch import optim as optim

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch import pytorch_util as ptu
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.algos.util import np_to_pytorch_batch
from railrl.torch.data_management.normalizer import TorchNormalizer


class Dagger(TorchRLAlgorithm):
    def __init__(
            self,
            env,
            model,
            mpc_controller,
            learning_rate=1e-3,
            **kwargs
    ):
        # TODO: normalize input at beginning (maybe)
        super().__init__(
            env,
            mpc_controller,
            **kwargs
        )
        assert self.collection_mode == 'batch'
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

    def get_all_data(self):
        all_obs = self.replay_buffer._observations[:self.replay_buffer._top]
        all_actions = self.replay_buffer._actions[:self.replay_buffer._top]
        all_next_obs = self.replay_buffer._next_obs[:self.replay_buffer._top]
        return np_to_pytorch_batch(dict(
            all_obs=all_obs,
            all_actions=all_actions,
            all_next_obs=all_next_obs,
        ))

    def _do_training(self):
        batch = self.get_all_data()
        all_obs = batch['all_obs']
        all_actions = batch['all_actions']
        all_next_obs = batch['all_next_obs']

        losses = []
        num_batches = len(all_obs) // self.batch_size
        for bn in range(num_batches):
            slc = slice(bn*self.batch_size, (bn+1)*self.batch_size)
            obs = all_obs[slc, :]
            actions = all_actions[slc, :]
            next_obs = all_next_obs[slc, :]

            next_obs_delta_pred = self.model(obs, actions)
            next_obs_delta = next_obs - obs
            errors = (next_obs_delta_pred - next_obs_delta) ** 2
            loss = errors.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(ptu.get_numpy(loss))

        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
            self.eval_statistics.update(create_stats_ordered_dict(
                'Model Loss',
                losses,
                always_show_all_stats=True,
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Obs Deltas',
                ptu.get_numpy(next_obs_delta),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Predicted Obs Deltas',
                ptu.get_numpy(next_obs_delta_pred),
            ))

    def pretrain(self):
        pretrain_paths = []
        while len(pretrain_paths) < self.num_random_paths:
            pretrain_paths += self.eval_sampler.obtain_samples()
        ob_mean, ob_std, delta_mean, delta_std, ac_mean, ac_std = (
            compute_normalization(pretrain_paths)
        )
        delta_normalizer = TorchNormalizer(
            len(delta_mean), delta_mean, delta_std,
        )
        action_normalizer = TorchNormalizer(
            len(delta_mean), delta_mean, delta_std,
        )

    def _can_train(self):
        return (
            super()._can_train() and
            self.replay_buffer.num_steps_can_sample() // self.batch_size > 1
        )

    def _can_evaluate(self):
        return self.eval_statistics is not None

    @property
    def networks(self):
        return [
            self.model
        ]


def compute_normalization(paths):
    obs = np.vstack([path["observations"] for path in paths])
    next_obs = np.vstack([path["next_observations"] for path in paths])
    deltas = next_obs - obs
    actions = np.vstack([path["actions"] for path in paths])
    ob_mean = np.mean(obs, axis=0)
    ob_std = np.std(obs, axis=0)
    delta_mean = np.mean(deltas, axis=0)
    delta_std = np.std(deltas, axis=0)
    ac_mean = np.mean(actions, axis=0)
    ac_std = np.std(actions, axis=0)
    return ob_mean, ob_std, delta_mean, delta_std, ac_mean, ac_std
