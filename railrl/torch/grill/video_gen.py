import argparse
import json
import os
import os.path as osp
import uuid
from pathlib import Path

import joblib

from railrl.core import logger
from railrl.envs.vae_wrappers import VAEWrappedEnv, ConditionalVAEWrappedEnv
from railrl.pythonplusplus import find_key_recursive
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import set_gpu_mode

filename = str(uuid.uuid4())

import skvideo.io
import numpy as np
import time

import scipy.misc

from multiworld.core.image_env import ImageEnv
from railrl.core import logger
from railrl.envs.vae_wrappers import temporary_mode

class VideoSaveFunction:
    def __init__(self, env, variant):
        self.logdir = logger.get_snapshot_dir()
        self.save_period = variant.get('save_video_period', 50)
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        self.dump_video_kwargs['imsize'] = variant['imsize'] #env.imsize

        if "rows" not in self.dump_video_kwargs:
            self.dump_video_kwargs["rows"] = 2
        if "columns" not in self.dump_video_kwargs:
            self.dump_video_kwargs["columns"] = 5

        self.expl_env = None
        self.eval_env = None

        self.expl_data_collector = None
        self.eval_data_collector = None

        self.variant = variant
        self.state_based = variant.get("do_state_exp", False)

    def __call__(self, algo, epoch):
        if self.expl_env is None or self.eval_env is None:
            self.expl_env = algo.expl_env
            self.eval_env = algo.eval_env
            if self.state_based:
                self.expl_env = ImageEnv(
                    self.expl_env,
                    self.variant['imsize'],
                    init_camera=self.variant.get('init_camera', None),
                    transpose=True,
                    normalize=True,
                )
                self.eval_env = ImageEnv(
                    self.eval_env,
                    self.variant['imsize'],
                    init_camera=self.variant.get('init_camera', None),
                    transpose=True,
                    normalize=True,
                )

        if self.expl_data_collector is None or self.eval_data_collector is None:
            self.expl_data_collector = algo.expl_data_collector
            self.eval_data_collector = algo.eval_data_collector
            if self.state_based:
                import copy
                self.expl_data_collector = copy.deepcopy(self.expl_data_collector)
                self.expl_data_collector._env = self.expl_env
                self.expl_data_collector.end_epoch(-1)

                self.eval_data_collector = copy.deepcopy(self.eval_data_collector)
                self.eval_data_collector._env = self.eval_env
                self.eval_data_collector.end_epoch(-1)

        if self.state_based:
            max_path_length = self.variant['max_path_length']
            rows = self.dump_video_kwargs["rows"]
            columns = self.dump_video_kwargs["columns"]

            expl_paths = self.expl_data_collector.collect_new_paths(
                max_path_length=max_path_length,
                num_steps=max_path_length*rows*columns,
                discard_incomplete_paths=True,
            )
            self.expl_data_collector.end_epoch(-1)

            eval_paths = self.eval_data_collector.collect_new_paths(
                max_path_length=max_path_length,
                num_steps=max_path_length*rows*columns,
                discard_incomplete_paths=True,
            )
            self.eval_data_collector.end_epoch(-1)
        else:
            expl_paths = self.expl_data_collector.get_epoch_paths()
            eval_paths = self.eval_data_collector.get_epoch_paths()
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(self.logdir, 'video_{epoch}_expl.mp4'.format(epoch=epoch))
            dump_paths(self.expl_env,
                filename,
                expl_paths,
                "image_desired_goal" if self.state_based else "decoded_goal_image",
                **self.dump_video_kwargs,
            )

            filename = osp.join(self.logdir, 'video_{epoch}_eval.mp4'.format(epoch=epoch))
            dump_paths(self.eval_env,
                filename,
                eval_paths,
                "image_desired_goal",
                **self.dump_video_kwargs,
            )


def add_border(img, pad_length, pad_color, imsize=84):
    H = 3*imsize
    W = imsize
    img = img.reshape((3*imsize, imsize, -1))
    img2 = np.ones((H + 2 * pad_length, W + 2 * pad_length, img.shape[2]), dtype=np.uint8) * pad_color
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2


def get_image(goal, obs, recon_obs, imsize=84, pad_length=1, pad_color=255):
    if len(goal.shape) == 1:
        goal = goal.reshape(-1, imsize, imsize).transpose(2, 1, 0)
        obs = obs.reshape(-1, imsize, imsize).transpose(2,1,0)
        recon_obs = recon_obs.reshape(-1, imsize, imsize).transpose(2,1,0)
    img = np.concatenate((goal, obs, recon_obs))
    img = np.uint8(255 * img)
    if pad_length > 0:
        img = add_border(img, pad_length, pad_color)
    return img


def dump_video(
        env,
        policy,
        filename,
        rollout_function,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
):
    # num_channels = env.vae.input_channels
    num_channels = 1 if env.grayscale else 3
    frames = []
    H = 3*imsize
    W=imsize
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )
        # path = rollout_function(
        #     horizon,
        #     horizon,
        #     discard_incomplete_paths=True,
        # )[0]
        is_vae_env = isinstance(env, VAEWrappedEnv)
        is_conditional_vae_env = isinstance(env, ConditionalVAEWrappedEnv)

        l = []
        x_0 = path['full_observations'][0]['image_observation']
        for d in path['full_observations']:
            if is_conditional_vae_env:
                recon = np.clip(env._reconstruct_img(d['image_observation'], x_0), 0, 1)
            elif is_vae_env:
                recon = np.clip(env._reconstruct_img(d['image_observation']), 0, 1)
            else:
                recon = d['image_observation']
            l.append(
                get_image(
                    d.get('decoded_goal_image', d['image_desired_goal']),
                    d['image_observation'],
                    recon,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    imsize=imsize,
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir+"/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir+"/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir+"/"+str(j)+".png", img)
        if do_timer:
            print(i, time.time() - start)

    frames = np.array(frames, dtype=np.uint8)
    path_length = frames.size // (
            N * (H + 2*pad_length) * (W + 2*pad_length) * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k+1, :, :, :, :].reshape(
                (path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)


def dump_paths(
        env,
        filename,
        paths,
        goal_image_key,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
):
    # num_channels = env.vae.input_channels
    num_channels = 1 if env.grayscale else 3
    frames = []
    H = 3 * imsize
    W = imsize
    rows = min(rows, int(len(paths) / columns))
    N = rows * columns
    is_vae_env = isinstance(env, VAEWrappedEnv)
    is_conditional_vae_env = isinstance(env, ConditionalVAEWrappedEnv)
    for i in range(N):
        start = time.time()
        path = paths[i]
        l = []
        x_0 = path['full_observations'][0]['image_observation']
        for d in path['full_observations']:
            if is_conditional_vae_env:
                recon = np.clip(env._reconstruct_img(d['image_observation'], x_0), 0, 1)
            elif is_vae_env:
                recon = np.clip(env._reconstruct_img(d['image_observation']), 0, 1)
            else:
                recon = d['image_observation']
            l.append(
                get_image(
                    d[goal_image_key],
                    d['image_observation'],
                    recon,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    imsize=imsize,
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir+"/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir+"/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir+"/"+str(j)+".png", img)
        if do_timer:
            print(i, time.time() - start)

    frames = np.array(frames, dtype=np.uint8)
    path_length = frames.size // (
            N * (H + 2*pad_length) * (W + 2*pad_length) * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k+1, :, :, :, :].reshape(
                (path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)

