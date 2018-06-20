from railrl.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
import railrl.torch.pytorch_util as ptu
import numpy as np
import torch

class OnlineVaeRelabelingBuffer(ObsDictRelabelingBuffer):

    def __init__(
        self,
        vae,
        *args,
        vae_obs_key='image_observation',
        vae_achieved_goal_key='image_achieved_goal',
        vae_desired_goal_key='image_desired_goal',
        exploration_rewards_type='None',
        ob_keys_to_save=None,
        **kwargs
    ):
        self.vae = vae
        self.vae_obs_key = vae_obs_key
        self.vae_desired_goal_key = vae_desired_goal_key
        self.vae_achieved_goal_key = vae_achieved_goal_key
        self.exploration_rewards_type = exploration_rewards_type

        if ob_keys_to_save is None:
            ob_keys_to_save = ['observation', 'desired_goal', 'achieved_goal']

        extra_keys = [self.vae_obs_key, self.vae_achieved_goal_key, self.vae_desired_goal_key]
        for key in extra_keys:
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)
        super().__init__(ob_keys_to_save=ob_keys_to_save, *args, **kwargs)

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)

        batch_idxs = batch['indices'].flatten()
        vae_obs = self._next_obs[self.vae_obs_key][batch_idxs]

        batch['exploration_rewards'] = {
            'reconstruction_error': self.reconstruction_mse,
            'latent_distance':      self.latent_novelty,
            'None':                 self.no_reward,
        }[self.exploration_rewards_type](vae_obs)

        batch['exploration_rewards'].reshape(-1, 1)
        return batch

    def random_vae_training_data(self, batch_size):
        indices = self._sample_indices(batch_size)
        vae_data = self._next_obs[self.vae_obs_key][indices]
        return ptu.np_to_var(vae_data)

    def refresh_latents(self, epoch):
        batch_size = 512
        next_idx = min(batch_size, self._size)
        cur_idx = 0
        while cur_idx < self._size:
            idxs = np.arange(cur_idx, next_idx)
            self._obs[self.observation_key][idxs] = \
                    self.env._encode(self._obs[self.vae_obs_key][idxs])
            self._next_obs[self.observation_key][idxs] = \
                    self.env._encode(self._next_obs[self.vae_obs_key][idxs])
            self._next_obs[self.desired_goal_key][idxs] = \
                    self.env._encode(self._next_obs[self.vae_desired_goal_key][idxs])
            self._next_obs[self.achieved_goal_key][idxs] = \
                    self.env._encode(self._next_obs[self.vae_achieved_goal_key][idxs])

            cur_idx += batch_size
            next_idx += batch_size
            next_idx = min(next_idx, self._size)

    def reconstruction_mse(self, vae_obs):
        n_samples = len(vae_obs)
        torch_input = ptu.np_to_var(vae_obs)
        recon_vae_obs, _, _ = self.vae(torch_input)

        error = torch_input - recon_vae_obs
        mse = torch.sum(error**2, dim=1) / n_samples
        return ptu.get_numpy(mse)

    def latent_novelty(self, vae_obs):
        distances = (self.env._encode(vae_obs) - self.vae.dist_mu)**2 / self.vae.dist_std
        return distances.sum(axis=1)

    def no_reward(self, vae_obs):
        return np.zeros(vae_obs.shape)
