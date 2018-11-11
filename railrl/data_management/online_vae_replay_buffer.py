from railrl.data_management.obs_dict_replay_buffer import flatten_dict
from railrl.data_management.shared_obs_dict_replay_buffer import \
        SharedObsDictRelabelingBuffer
from multiworld.core.image_env import normalize_image
import railrl.torch.pytorch_util as ptu
import numpy as np
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from railrl.torch.networks import Mlp
from railrl.misc.ml_util import ConstantSchedule
from railrl.misc.ml_util import PiecewiseLinearSchedule
from railrl.torch.vae.vae_trainer import (
    inv_gaussian_p_x_np_to_np,
    inv_p_bernoulli_x_np_to_np,
    compute_inv_exp_elbo)


class OnlineVaeRelabelingBuffer(SharedObsDictRelabelingBuffer):

    def __init__(
        self,
        vae,
        *args,
        decoded_obs_key='image_observation',
        decoded_achieved_goal_key='image_achieved_goal',
        decoded_desired_goal_key='image_desired_goal',
        exploration_rewards_type='None',
        exploration_rewards_scale=1.0,
        vae_priority_type='None',
        power=1.0,
        internal_keys=None,
        exploration_schedule_kwargs=None,
        priority_function_kwargs=None,
        **kwargs
    ):
        self.quick_init(locals())
        self.vae = vae
        self.decoded_obs_key = decoded_obs_key
        self.decoded_desired_goal_key = decoded_desired_goal_key
        self.decoded_achieved_goal_key = decoded_achieved_goal_key
        self.exploration_rewards_type = exploration_rewards_type
        self.exploration_rewards_scale = exploration_rewards_scale
        self.vae_priority_type = vae_priority_type
        self.power = power

        if exploration_schedule_kwargs is None:
            self.explr_reward_scale_schedule = \
                    ConstantSchedule(self.exploration_rewards_scale)
        else:
            self.explr_reward_scale_schedule = \
                    PiecewiseLinearSchedule(**exploration_schedule_kwargs)

        extra_keys = [
            self.decoded_obs_key,
            self.decoded_achieved_goal_key,
            self.decoded_desired_goal_key
        ]
        if internal_keys is None:
            internal_keys = []

        for key in extra_keys:
            if key in internal_keys:
                continue
            internal_keys.append(key)
        super().__init__(internal_keys=internal_keys, *args, **kwargs)
        self._give_explr_reward_bonus = (
            exploration_rewards_type != 'None'
            and exploration_rewards_scale != 0.
        )
        self._exploration_rewards = np.zeros((self.max_size, 1))
        self._prioritize_vae_samples = (
            vae_priority_type != 'None'
            and power != 0.
        )
        self._vae_sample_priorities = np.zeros((self.max_size, 1))
        self._vae_sample_probs = None

        self.use_dynamics_model = (
            self.exploration_rewards_type == 'forward_model_error'
        )
        if self.use_dynamics_model:
            self.initialize_dynamics_model()

        type_to_function = {
            'reconstruction_error':         self.reconstruction_mse,
            'bce':                          self.binary_cross_entropy,
            'latent_distance':              self.latent_novelty,
            'latent_distance_true_prior':   self.latent_novelty_true_prior,
            'forward_model_error':          self.forward_model_error,
            'gaussian_inv_prob':            self.gaussian_inv_prob,
            'bernoulli_inv_prob':           self.bernoulli_inv_prob,
            'image_gaussian_inv_prob':      self.image_gaussian_inv_prob,
            'image_bernoulli_inv_prob':     self.image_bernoulli_inv_prob,
            'inv_exp_elbo':                 self.inv_exp_elbo,
            'None':                         self.no_reward,
        }

        self.exploration_reward_func = (
            type_to_function[self.exploration_rewards_type]
        )
        self.vae_prioritization_func = (
            type_to_function[self.vae_priority_type]
        )

        if priority_function_kwargs is None:
            self.priority_function_kwargs = dict()
        else:
            self.priority_function_kwargs = priority_function_kwargs

        self.epoch = 0
        self._register_mp_array("_exploration_rewards")
        self._register_mp_array("_vae_sample_priorities")

    def add_path(self, path):
        self.add_decoded_vae_goals_to_path(path)
        super().add_path(path)

    def add_decoded_vae_goals_to_path(self, path):
        # decoding the self-sampled vae images should be done in batch (here)
        # rather than in the env for efficiency
        desired_goals = flatten_dict(
            path['observations'],
            [self.desired_goal_key]
        )[self.desired_goal_key]
        desired_decoded_goals = self.env._decode(desired_goals)
        desired_decoded_goals = desired_decoded_goals.reshape(
            len(desired_decoded_goals),
            -1
        )
        for idx, next_obs in enumerate(path['observations']):
            path['observations'][idx][self.decoded_desired_goal_key] = \
                    desired_decoded_goals[idx]
            path['next_observations'][idx][self.decoded_desired_goal_key] = \
                    desired_decoded_goals[idx]

    def random_batch(self, batch_size):
        batch = super().random_batch(batch_size)
        exploration_rewards_scale = float(self.explr_reward_scale_schedule.get_value(self.epoch))
        if self._give_explr_reward_bonus:
            batch_idxs = batch['indices'].flatten()
            batch['exploration_rewards'] = self._exploration_rewards[batch_idxs]
            batch['rewards'] += exploration_rewards_scale * batch['exploration_rewards']
        return batch

    def refresh_latents(self, epoch):
        self.epoch = epoch
        batch_size = 1024
        next_idx = min(batch_size, self._size)
        cur_idx = 0
        while cur_idx < self._size:
            idxs = np.arange(cur_idx, next_idx)
            self._obs[self.observation_key][idxs] = \
                self.env._encode(
                    normalize_image(self._obs[self.decoded_obs_key][idxs])
                )
            self._next_obs[self.observation_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.decoded_obs_key][idxs])
                )
            self._next_obs[self.desired_goal_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.decoded_desired_goal_key][idxs])
                )
            self._next_obs[self.achieved_goal_key][idxs] = \
                self.env._encode(
                    normalize_image(self._next_obs[self.decoded_achieved_goal_key][idxs])
                )
            normalized_imgs = (
                normalize_image(self._next_obs[self.decoded_obs_key][idxs])
            )
            if self._give_explr_reward_bonus:
                rewards = self.exploration_reward_func(
                    normalized_imgs,
                    idxs,
                    **self.priority_function_kwargs
                )
                self._exploration_rewards[idxs] = rewards.reshape(-1, 1)
            if self._prioritize_vae_samples:
                if (
                    self.exploration_rewards_type == self.vae_priority_type
                    and self._give_explr_reward_bonus
                ):
                    self._vae_sample_priorities[idxs] = (
                        self._exploration_rewards[idxs]
                    )
                else:
                    self._vae_sample_priorities[idxs] = (
                        self.vae_prioritization_func(
                            normalized_imgs,
                            idxs,
                            **self.priority_function_kwargs
                        ).reshape(-1, 1)
                    )

            cur_idx = next_idx
            next_idx += batch_size
            next_idx = min(next_idx, self._size)
        if self._prioritize_vae_samples:
            vae_sample_priorities = self._vae_sample_priorities[:self._size]
            self._vae_sample_probs = vae_sample_priorities ** self.power
            p_sum = np.sum(self._vae_sample_probs)
            assert p_sum > 0, "Unnormalized p sum is {}".format(p_sum)
            self._vae_sample_probs /= np.sum(self._vae_sample_probs)
            self._vae_sample_probs = self._vae_sample_probs.flatten()
            assert np.min(self._vae_sample_probs) > 0, "probs should not be 0"

    def random_vae_training_data(self, batch_size):
        if self._prioritize_vae_samples and self._vae_sample_probs is not None:
            indices = np.random.choice(
                len(self._vae_sample_probs),
                batch_size,
                p=self._vae_sample_probs,
            )
        else:
            indices = self._sample_indices(batch_size)

        next_obs = normalize_image(self._next_obs[self.decoded_obs_key][indices])
        return dict(
            next_obs=ptu.from_numpy(next_obs)
        )

    def reconstruction_mse(self, next_vae_obs, indices):
        torch_input = ptu.from_numpy(next_vae_obs)
        recon_next_vae_obs, _, _ = self.vae(torch_input)

        error = torch_input - recon_next_vae_obs
        mse = torch.sum(error**2, dim=1)
        return ptu.get_numpy(mse)

    def gaussian_inv_prob(self, next_vae_obs, indices):
        return np.exp(self.reconstruction_mse(next_vae_obs, indices))

    def binary_cross_entropy(self, next_vae_obs, indices):
        torch_input = ptu.from_numpy(next_vae_obs)
        recon_next_vae_obs, _, _ = self.vae(torch_input)

        error = - torch_input * torch.log(
            torch.clamp(
                recon_next_vae_obs,
                min=1e-30,  # corresponds to about -70
            )
        )
        bce = torch.sum(error, dim=1)
        return ptu.get_numpy(bce)

    def bernoulli_inv_prob(self, next_vae_obs, indices):
        torch_input = ptu.np_to_var(next_vae_obs)
        recon_next_vae_obs, _, _ = self.vae(torch_input)
        prob = (
            torch_input * recon_next_vae_obs
            + (1 - torch_input) * (1 - recon_next_vae_obs)
        ).prod(dim=1)
        return ptu.get_numpy(1 / prob)

    def image_gaussian_inv_prob(self, next_vae_obs, indices, num_latents_to_sample=1):
        return inv_gaussian_p_x_np_to_np(self.vae, next_vae_obs, num_latents_to_sample=num_latents_to_sample)

    def image_bernoulli_inv_prob(self, next_vae_obs, indices, num_latents_to_sample=1):
        return inv_p_bernoulli_x_np_to_np(self.vae, next_vae_obs, num_latents_to_sample=num_latents_to_sample)

    def inv_exp_elbo(self, next_vae_obs, indices, beta):
        pass

    def forward_model_error(self, next_vae_obs, indices):
        obs = self._obs[self.observation_key][indices]
        next_obs = self._next_obs[self.observation_key][indices]
        actions = self._actions[indices]

        state_action_pair = ptu.from_numpy(np.c_[obs, actions])
        prediction = self.dynamics_model(state_action_pair)
        mse = self.dynamics_loss(prediction, ptu.from_numpy(next_obs))
        return ptu.get_numpy(mse)

    def latent_novelty(self, next_vae_obs, indices):
        distances = ((self.env._encode(next_vae_obs) - self.vae.dist_mu) /
                     self.vae.dist_std)**2
        return distances.sum(axis=1)

    def latent_novelty_true_prior(self, next_vae_obs, indices):
        distances = self.env._encode(next_vae_obs)**2
        return distances.sum(axis=1)

    def _kl_np_to_np(self, next_vae_obs, indices):
        torch_input = ptu.np_to_var(next_vae_obs)
        mu, log_var = self.vae.encode(torch_input)
        return ptu.get_numpy(
            - torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        )

    def no_reward(self, next_vae_obs, indices):
        return np.zeros((len(next_vae_obs), 1))

    def initialize_dynamics_model(self):
        obs_dim = self._obs[self.observation_key].shape[1]
        self.dynamics_model = Mlp(
            hidden_sizes=[128, 128],
            output_size=obs_dim,
            input_size=obs_dim + self._action_dim,
        )
        if ptu.gpu_enabled():
            self.dynamics_model.to(ptu.device)
        self.dynamics_optimizer = Adam(self.dynamics_model.parameters())
        self.dynamics_loss = MSELoss()

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

            state_action_pair = ptu.from_numpy(np.c_[obs, actions])
            prediction = self.dynamics_model(state_action_pair)
            mse = self.dynamics_loss(prediction, ptu.from_numpy(next_obs))

            mse.backward()
            self.dynamics_optimizer.step()
