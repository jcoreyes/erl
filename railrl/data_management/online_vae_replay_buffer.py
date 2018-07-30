from railrl.data_management.obs_dict_replay_buffer import \
        ObsDictRelabelingBuffer, flatten_dict
from multiworld.core.image_env import normalize_image
import railrl.torch.pytorch_util as ptu
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from railrl.torch.networks import Mlp

class OnlineVaeRelabelingBuffer(ObsDictRelabelingBuffer):

    def __init__(
        self,
        vae,
        *args,
        vae_obs_key='image_observation',
        vae_achieved_goal_key='image_achieved_goal',
        vae_desired_goal_key='image_desired_goal',
        exploration_rewards_type='None',
        exploration_rewards_scale=1.0,
        alpha=1.0,
        ob_keys_to_save=None,
        **kwargs
    ):
        self.vae = vae
        self.vae_obs_key = vae_obs_key
        self.vae_desired_goal_key = vae_desired_goal_key
        self.vae_achieved_goal_key = vae_achieved_goal_key
        self.exploration_rewards_type = exploration_rewards_type
        self.exploration_rewards_scale = exploration_rewards_scale

        if ob_keys_to_save is None:
            ob_keys_to_save = ['observation', 'desired_goal', 'achieved_goal']

        extra_keys = [
            self.vae_obs_key,
            self.vae_achieved_goal_key,
            self.vae_desired_goal_key
        ]
        for key in extra_keys:
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)
        super().__init__(ob_keys_to_save=ob_keys_to_save, *args, **kwargs)
        self._exploration_rewards = np.zeros((self.max_size, 1))

        self.total_exploration_error = 0.0
        self.alpha = alpha
        self.use_dynamics_model = \
                self.exploration_rewards_type == 'forward_model_error' or \
                self.exploration_rewards_type == 'inverse_model_error'
        if self.use_dynamics_model:
            self.initialize_dynamics_model()

        self.exploration_reward_func = {
            'reconstruction_error': self.reconstruction_mse,
            'latent_distance':      self.latent_novelty,
            'forward_model_error':  self.forward_model_error,
            'inverse_model_error':  self.inverse_model_error,
            'None':                 self.no_reward,
        }[self.exploration_rewards_type]

    def add_path(self, path):
        self.add_decoded_vae_to_path(path, 'observations')
        self.add_decoded_vae_to_path(path, 'next_observations')
        super().add_path(path)

    def add_decoded_vae_to_path(self, path, key):
        # decoding the self-sampled vae image should be done in batch.
        desired_goals = flatten_dict(
            path[key],
            [self.desired_goal_key]
        )[self.desired_goal_key]
        vae_desired_goals = self.env._decode(desired_goals)
        vae_desired_goals = vae_desired_goals.reshape(len(vae_desired_goals), -1)
        for idx, next_obs in enumerate(path[key]):
            path[key][idx][self.vae_desired_goal_key] = vae_desired_goals[idx]

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)
        batch_idxs = batch['indices'].flatten()
        vae_obs = self._next_obs[self.vae_obs_key][batch_idxs]
        batch['exploration_rewards'] = self._exploration_rewards[batch_idxs]
        if self.use_dynamics_model:
            print('before', max(batch['rewards']))
            batch['rewards'] += self.exploration_rewards_scale * \
                    self.dynamics_model_error(vae_obs, batch_idxs)
            print('after', max(batch['rewards']))
        else:
            batch['rewards'] += \
                    self.exploration_rewards_scale * batch['exploration_rewards']
        return batch

    def refresh_latents(self, epoch):
        batch_size = 1024
        next_idx = min(batch_size, self._size)
        cur_idx = 0
        idxs = np.arange(cur_idx, next_idx)
        while cur_idx < self._size:
            idxs = np.arange(cur_idx, next_idx)
            self._obs[self.observation_key][idxs] = \
                self.env._encode(
                    normalize_image(self._obs[self.vae_obs_key][idxs])
                )
            self._next_obs[self.observation_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.vae_obs_key][idxs])
                )
            self._next_obs[self.desired_goal_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.vae_desired_goal_key][idxs])
                )
            self._next_obs[self.achieved_goal_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.vae_achieved_goal_key][idxs])
                )
            self._exploration_rewards[idxs] = self.exploration_reward_func(
                normalize_image(self._next_obs[self.vae_obs_key][idxs]),
                idxs,
            ).reshape(-1, 1)

            cur_idx += batch_size
            next_idx += batch_size
            next_idx = min(next_idx, self._size)
        self.total_exploration_error = np.sum(self._exploration_rewards[:self._size])
        self.vae_probs = self._exploration_rewards[:self._size]**self.alpha
        if np.sum(self.vae_probs) != 0.0:
            self.vae_probs /= np.sum(self.vae_probs)
        self.vae_probs = self.vae_probs.flatten()

    def random_vae_training_data(self, batch_size):
        indices = self._sample_indices(batch_size)
        if self.total_exploration_error != 0.0:
            indices = np.random.choice(
                len(self.vae_probs),
                batch_size,
                p=self.vae_probs,
            )
        next_obs = normalize_image(self._next_obs[self.vae_obs_key][indices])
        obs = normalize_image(self._obs[self.vae_obs_key][indices])
        actions = self._actions[indices]
        return dict(
            obs=ptu.np_to_var(obs),
            next_obs=ptu.np_to_var(next_obs),
            actions=ptu.np_to_var(actions),
        )

    def reconstruction_mse(self, next_vae_obs, indices):
        n_samples = len(next_vae_obs)
        torch_input = ptu.np_to_var(next_vae_obs)
        recon_next_vae_obs, _, _ = self.vae(torch_input)

        error = torch_input - recon_next_vae_obs
        mse = torch.sum(error**2, dim=1) / n_samples
        return ptu.get_numpy(mse)

    def forward_model_error(self, next_vae_obs, indices):
        obs = self._obs[self.observation_key][indices]
        next_obs = self._next_obs[self.observation_key][indices]
        actions = self._actions[indices]

        state_action_pair = ptu.np_to_var(np.c_[obs, actions])
        prediction = self.dynamics_model(state_action_pair)
        mse = self.dynamics_loss(prediction, ptu.np_to_var(next_obs))
        return ptu.get_numpy(mse)

    def inverse_model_error(self, next_vae_obs, indices):
        obs = self._obs[self.observation_key][indices]
        next_obs = self._next_obs[self.observation_key][indices]
        obs, next_obs = next_obs, obs
        actions = self._actions[indices]

        state_action_pair = ptu.np_to_var(np.c_[obs, actions])
        prediction = self.dynamics_model(state_action_pair)
        mse = self.dynamics_loss(prediction, ptu.np_to_var(next_obs))
        return ptu.get_numpy(mse)

    def latent_novelty(self, next_vae_obs, indices):
        distances = (self.env._encode(vae_obs) - self.vae.dist_mu / self.vae.dist_std)**2
        return distances.sum(axis=1)

    def no_reward(self, next_vae_obs, indices):
        return np.zeros((len(vae_obs), 1))

    def initialize_dynamics_model(self):
        self.dynamics_model = Mlp(
            hidden_sizes=[128, 128],
            output_size=self._obs[self.observation_key].shape[1],
            input_size=self._obs[self.observation_key].shape[1] + self._action_dim,
        )
        if ptu.gpu_enabled():
            self.dynamics_model.cuda()
        self.dynamics_optimizer = Adam(self.dynamics_model.parameters())
        self.dynamics_loss = MSELoss()
        if self.exploration_rewards_type == 'forward_model_error':
            self.dynamics_model_error = self.forward_model_error
        else:
            self.dynamics_model_error = self.inverse_model_error

    def train_dynamics_model(self, batches=50, batch_size=100):
        if not self.use_dynamics_model:
            return
        for _ in range(batches):
            indices = self._sample_indices(batch_size)
            self.dynamics_optimizer.zero_grad()
            obs = self._obs[self.observation_key][indices]
            next_obs = self._next_obs[self.observation_key][indices]
            actions = self._actions[indices]
            if self.exploration_rewards_type == 'inverse_model_error':
                obs, next_obs = next_obs, obs

            state_action_pair = ptu.np_to_var(np.c_[obs, actions])
            prediction = self.dynamics_model(state_action_pair)
            mse = self.dynamics_loss(prediction, ptu.np_to_var(next_obs))

            mse.backward()
            self.dynamics_optimizer.step()
