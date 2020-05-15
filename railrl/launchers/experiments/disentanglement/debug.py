import time
from collections import OrderedDict, namedtuple
from os import path as osp
from typing import List
import typing

import cv2
import gym
import numpy as np
from torch import optim

from railrl.core import logger
from railrl.envs.images import Renderer, InsertImagesEnv
from railrl.torch import pytorch_util as ptu
from railrl.torch.core import eval_np
from railrl.torch.disentanglement.networks import DisentangledMlpQf
from railrl.torch.networks import Mlp
from railrl.torch.torch_rl_algorithm import TorchTrainer
from railrl.visualization.image import combine_images_into_grid


class JointTrainer(TorchTrainer):
    def __init__(self, trainers: List[TorchTrainer]):
        super().__init__()
        if len(trainers) == 0:
            raise ValueError("Need at least one trainer")
        self._trainers = trainers

    def train_from_torch(self, batch):
        for trainer in self._trainers:
            trainer.train_from_torch(batch)

    @property
    def networks(self):
        for trainer in self._trainers:
            for net in trainer.networks:
                yield net

    def end_epoch(self, epoch):
        for trainer in self._trainers:
            trainer.end_epoch(epoch)

    def get_snapshot(self):
        snapshot = self._trainers[0].get_snapshot()
        for trainer in self._trainers[1:]:
            snapshot.update(trainer.get_snapshot())
        return snapshot

    def get_diagnostics(self):
        stats = self._trainers[0].get_diagnostics()
        for trainer in self._trainers[1:]:
            stats.update(trainer.get_diagnostics())
        return stats


class DebugTrainer(TorchTrainer):
    def __init__(self, observation_space, encoder, encoder_output_dim):
        super().__init__()
        self._ob_space = observation_space
        self._encoder = encoder
        self._latent_dim = encoder_output_dim

    def train_from_torch(self, batch):
        pass

    @property
    def networks(self):
        return []

    def get_diagnostics(self):
        start_time = time.time()
        linear_loss = get_linear_loss(self._ob_space, self._encoder)
        linear_time = time.time() - start_time

        start_time = time.time()
        non_linear_results = get_non_linear_results(
            self._ob_space, self._encoder, self._latent_dim)
        non_linear_time = time.time() - start_time

        stats = OrderedDict([
            ('debug/reconstruction/linear/loss', linear_loss),
            ('debug/reconstruction/linear/train_time (s)', linear_time),
            ('debug/reconstruction/non_linear/loss', non_linear_results.loss),
            ('debug/reconstruction/non_linear/initial_loss',
             non_linear_results.initial_loss),
            ('debug/reconstruction/non_linear/last_10_percent_contribution',
             non_linear_results.last_10_percent_contribution),
            ('debug/reconstruction/non_linear/train_time (s)', non_linear_time),
        ])
        return stats


def get_linear_loss(ob_space, encoder):
    x = get_batch(ob_space, batch_size=2 ** 15)
    z_np = eval_np(encoder, x)
    results = np.linalg.lstsq(z_np, x, rcond=None)
    matrix = results[0]

    eval_states = get_batch(ob_space, batch_size=2 ** 15)
    z_np = eval_np(encoder, eval_states)
    x_hat = z_np.dot(matrix)
    return ((eval_states - x_hat) ** 2).mean()


NonLinearResults = namedtuple(
    'NonLinearResults',
    [
        'loss',
        'last_10_percent_contribution',
        'initial_loss',
    ],
)


def get_non_linear_results(
    ob_space, encoder, latent_dim,
    batch_size=128,
    num_batches=10000,
) -> NonLinearResults:
    state_dim = ob_space.low.size

    decoder = Mlp(
        hidden_sizes=[64, 64],
        output_size=state_dim,
        input_size=latent_dim,
    )
    decoder.to(ptu.device)
    optimizer = optim.Adam(decoder.parameters())

    initial_loss = last_10_percent_loss = 0
    for i in range(num_batches):
        states = get_batch(ob_space, batch_size)
        x = ptu.from_numpy(states)
        z = encoder(x)
        x_hat = decoder(z)

        loss = ((x - x_hat) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i == 0:
            initial_loss = ptu.get_numpy(loss)
        if i == int(num_batches * 0.9):
            last_10_percent_loss = ptu.get_numpy(loss)

    eval_states = get_batch(ob_space, batch_size=2 ** 15)
    x = ptu.from_numpy(eval_states)
    z = encoder(x)
    x_hat = decoder(z)
    reconstruction = ptu.get_numpy(x_hat)
    loss = ((eval_states - reconstruction) ** 2).mean()
    last_10_percent_contribution = (
                                       (last_10_percent_loss - loss)
                                   ) / (initial_loss - loss)
    del decoder, optimizer
    return NonLinearResults(
        loss=loss,
        initial_loss=initial_loss,
        last_10_percent_contribution=last_10_percent_contribution,
    )


def get_batch(ob_space, batch_size):
    noise = np.random.randn(batch_size, *ob_space.low.shape)
    return noise * (ob_space.high - ob_space.low) + ob_space.low


class DebugRenderer(Renderer):
    def __init__(
            self,
            encoder: DisentangledMlpQf,
            head_index,
            **kwargs
    ):
        super().__init__(**kwargs)
        """Render an image."""
        self.channels = 3
        self.encoder = encoder
        self.head_idx = head_index

    def create_image(self, env, encoded):
        values = encoded[:, self.head_idx]
        value_image = values.reshape(self.image_shape[:2])
        value_img_rgb = np.repeat(
            value_image[:, :, None],
            3,
            axis=2
        )
        value_img_rgb = (
                (value_img_rgb - value_img_rgb.min()) /
                (value_img_rgb.max() - value_img_rgb.min())
        )
        return value_img_rgb

class InsertDebugImagesEnv(InsertImagesEnv):
    def __init__(
            self,
            wrapped_env: gym.Env,
            renderers: typing.Dict[str, DebugRenderer],
            compute_shared_data=None,
    ):
        super().__init__(wrapped_env, renderers)
        self.compute_shared_data = compute_shared_data

    def _update_obs(self, obs):
        shared_data = self.compute_shared_data(obs, self.env)
        for image_key, renderer in self.renderers.items():
            obs[image_key] = renderer.create_image(self.env, shared_data)


def create_visualize_representation(encoder, sweep_object_zero, env, renderer,
        save_period=50, num_presampled_states=2, num_random_states=2,
        env_renderer=None,
        initial_save_period=None,
                                    state_to_encoder_input=None):
    if initial_save_period is None:
        initial_save_period = save_period
    env_renderer = env_renderer or renderer
    state_space = env.env.observation_space['state_observation']
    low = state_space.low.min()
    high = state_space.high.max()
    y = np.linspace(low, high, num=env_renderer.image_chw[1])
    x = np.linspace(low, high, num=env_renderer.image_chw[2])
    all_xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    # y = np.linspace(low / 2, high / 2, num=2)
    # x = np.linspace(low / 2, high / 2, num=2)
    # all_start_xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    # same_start_state = np.vstack(
    #     [np.hstack([xy, xy]) for xy in all_start_xy]
    # )
    same_start_state = np.vstack([
        state_space.sample() for _ in range(num_presampled_states)
    ])
    random_states = np.vstack([
        state_space.sample() for _ in range(num_random_states)
    ])
    all_start_xy = np.concatenate((same_start_state, random_states), axis=0)

    goal_dicts = []
    for start_state in all_start_xy:
        if sweep_object_zero:
            new_states = np.concatenate(
                [
                    all_xy,
                    np.repeat(start_state[None, 2:], all_xy.shape[0], axis=0),
                ],
                axis=1,
            )
        else:
            new_states = np.concatenate(
                [
                    np.repeat(start_state[None, :2], all_xy.shape[0], axis=0),
                    all_xy,
                ],
                axis=1,
            )
        if state_to_encoder_input:
            img_obs = True
            new_states = np.concatenate(
                [
                    state_to_encoder_input(state)[None, ...] for state in new_states
                ],
                axis=0,
            )
        else:
            img_obs = False
        goal_dict = {
            'state_desired_goal': start_state,
        }
        env_state = env.get_env_state()
        env.set_to_goal(goal_dict)
        start_img = renderer.create_image(env)
        env.set_env_state(env_state)
        goal_dict['image_observation'] = start_img
        goal_dict['new_states'] = new_states
        goal_dicts.append(goal_dict)

    def visualize_representation(algo, epoch):
        if (
                (epoch < save_period and epoch % initial_save_period == 0)
                or epoch % save_period == 0
                or epoch >= algo.num_epochs - 1
        ):
            logdir = logger.get_snapshot_dir()
            if sweep_object_zero:
                filename = osp.join(
                    logdir,
                    'obj0_sweep_visualization_{epoch}.png'.format(epoch=epoch),
                )
            else:
                filename = osp.join(
                    logdir,
                    'obj1_sweep_visualization_{epoch}.png'.format(epoch=epoch),
                )

            columns = []
            for goal_dict in goal_dicts:
                start_img = goal_dict['image_observation']
                new_states = goal_dict['new_states']

                encoded = encoder.encode(new_states)
                # img_format = renderer.output_image_format
                images_to_stack = [start_img]
                for i in range(encoded.shape[1]):
                    values = encoded[:, i]
                    value_image = values.reshape(env_renderer.image_chw[1:])
                    # TODO: fix hardcoding of CHW
                    value_img_rgb = np.repeat(
                        value_image[None, :, :],
                        3,
                        axis=0
                    )
                    value_img_rgb = (
                            (value_img_rgb - value_img_rgb.min()) /
                            (value_img_rgb.max() - value_img_rgb.min() + 1e-9)
                    )
                    images_to_stack.append(value_img_rgb)

                columns.append(
                    combine_images_into_grid(
                        images_to_stack,
                        imwidth=renderer.image_chw[2],
                        imheight=renderer.image_chw[1],
                        max_num_cols=5,
                        pad_length=1,
                        pad_color=0,
                        subpad_length=1,
                        subpad_color=128,
                        image_format=renderer.output_image_format,
                    )
                )

            final_image = np.concatenate(columns, axis=1)
            cv2.imwrite(filename, final_image)

            print("Saved visualization image to to ", filename)

    return visualize_representation
