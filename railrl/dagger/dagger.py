from collections import OrderedDict

from torch import optim as optim

from railrl.misc.data_processing import create_stats_ordered_dict
from railrl.torch import pytorch_util as ptu
from railrl.torch.algos.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.algos.util import np_to_pytorch_batch


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
