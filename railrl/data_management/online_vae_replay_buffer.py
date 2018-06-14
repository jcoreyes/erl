from railrl.data_management.obs_dict_replay_buffer import ObsDictRelabelingBuffer
import railrl.torch.pytorch_util as ptu
import numpy as np

class OnlineVaeRelabelingBuffer(ObsDictRelabelingBuffer):

    def __init__(
        self,
        vae,
        *args,
        vae_obs_key='image_observation',
        vae_goal_key='image_desired_goal',
        ob_keys_to_save=None,
        **kwargs
    ):
        self.vae = vae
        self.vae_obs_key = vae_obs_key
        self.vae_goal_key = vae_goal_key

        if ob_keys_to_save is None:
            ob_keys_to_save = ['observation', 'desired_goal', 'achieved_goal']

        extra_keys = [self.vae_obs_key, self.vae_goal_key]
        for key in extra_keys:
            if key not in ob_keys_to_save:
                ob_keys_to_save.append(key)

        super().__init__(ob_keys_to_save=ob_keys_to_save, *args, **kwargs)

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
                    self.env._encode(self._next_obs[self.vae_goal_key][idxs])

            cur_idx += batch_size
            next_idx += batch_size
            next_idx = min(next_idx, self._size)



