import argparse
import json
import os
import os.path as osp
import uuid
from pathlib import Path

import joblib

from railrl.core import logger
from railrl.pythonplusplus import find_key_recursive
from railrl.samplers.rollout_functions import tdm_rollout
from railrl.torch.core import PyTorchModule
from railrl.torch.pytorch_util import set_gpu_mode

filename = str(uuid.uuid4())

import skvideo.io
import numpy as np
import time

import scipy.misc

H = 168
W = 84
PAD = 0 # False
if PAD:
    W += 2 * PAD
    H += 2 * PAD


def add_border(img):
    img = img.reshape((168, 84, -1))
    img2 = np.ones((H, W, img.shape[2]), dtype=np.uint8) * 255
    img2[PAD:-PAD, PAD:-PAD, :] = img
    return img2


def get_image(goal, obs):
    if len(goal.shape) == 1:
        goal = goal.reshape(3, 84, 84).transpose()
        obs = obs.reshape(3, 84, 84).transpose()
    img = np.concatenate((goal, obs))
    img = np.uint8(255 * img)
    if PAD:
        img = add_border(img)
    return img


def dump_video(
        env,
        policy,
        filename,
        rollout_function,
        ROWS=3,
        COLUMNS=6,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
):
    num_channels = env.vae.input_channels
    frames = []
    N = ROWS * COLUMNS
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            animated=False,
        )
        frames += [
            get_image(d['image_desired_goal'], d['image_observation'])
            for d in path['full_observations']
        ]
        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:84, :84, :], 0)
            scipy.misc.imsave(rollout_dir+"/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:84, :84, :], 0)
            scipy.misc.imsave(rollout_dir+"/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][84:, :84, :], 0)
                scipy.misc.imsave(rollout_dir+"/"+str(j)+".png", img)
        if do_timer:
            print(i, time.time() - start)

    frames = np.array(frames, dtype=np.uint8).reshape((N, horizon + 1, H, W, num_channels))
    f1 = []
    for k1 in range(COLUMNS):
        f2 = []
        for k2 in range(ROWS):
            k = k1 * ROWS + k2
            f2.append(frames[k:k+1, :, :, :, :].reshape((horizon + 1, H, W, num_channels)))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)
