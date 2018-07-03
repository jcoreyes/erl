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


def get_max_tau(args):
    if args.mtau is None:
        variant_path = Path(args.file).parents[0] / 'variant.json'
        variant = json.load(variant_path.open())
        max_tau = find_key_recursive(variant, 'max_tau')
        if max_tau is None:
            print("Defaulting max tau to 0.")
            max_tau = 0
        else:
            print("Max tau read from variant: {}".format(max_tau))
    else:
        max_tau = args.mtau
    return max_tau


def dump_video(
        env,
        policy,
        filename,
        ROWS=3,
        COLUMNS=6,
        do_timer=True,
        max_tau=0,
        horizon=100,
        dirname=None,
        subdirname="rollouts",
):
    policy.train(False) # is this right/necessary?
    paths = []
    num_channels = env.vae.input_channels
    frames = []
    N = ROWS * COLUMNS
    for i in range(N):
        rollout_dir = osp.join(dirname, subdirname, str(i))
        os.makedirs(rollout_dir, exist_ok=True)
        start = time.time()
        path = tdm_rollout(
            env,
            policy,
            init_tau=max_tau,
            max_path_length=horizon,
            animated=False,
            observation_key='latent_observation',
            desired_goal_key='latent_desired_goal',
        )
        frames += [
            get_image(d['image_desired_goal'], d['image_observation'])
            for d in path['full_observations']
        ]
        rollout_frames = frames[-101:]
        paths.append(path)
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

    return paths


def simulate_policy(args):
    data = joblib.load(args.file)
    if 'eval_policy' in data:
        policy = data['eval_policy']
    elif 'policy' in data:
        policy = data['policy']
    elif 'exploration_policy' in data:
        policy = data['exploration_policy']
    else:
        raise Exception("No policy found in loaded dict. Keys: {}".format(
            data.keys()
        ))
    max_tau = get_max_tau(args)

    env = data['env']

    env.mode("video_env")
    env.decode_goals = True

    if hasattr(env, 'enable_render'):
        # some environments need to be reconfigured for visualization
        env.enable_render()

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
        if hasattr(env, "vae"):
            env.vae.cuda()
    else:
        # make sure everything is on the CPU
        set_gpu_mode(False)
        policy.cpu()
        if hasattr(env, "vae"):
            env.vae.cpu()

    if args.pause:
        import ipdb; ipdb.set_trace()
    if isinstance(policy, PyTorchModule):
        policy.train(False)
    ROWS = 3
    COLUMNS = 6
    dirname = osp.dirname(args.file)
    input_file_name = os.path.splitext(
        os.path.basename(args.file)
    )[0]
    filename = osp.join(
        dirname, "video_{}.mp4".format(input_file_name)
    )
    paths = dump_video(
        env, policy, filename,
        max_tau=max_tau,
        ROWS=ROWS,
        COLUMNS=COLUMNS,
        horizon=args.H,
        dirname=dirname,
        subdirname="rollouts_" + input_file_name,
    )

    if hasattr(env, "log_diagnostics"):
        env.log_diagnostics(paths)
    logger.dump_tabular()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=100,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=10,
                        help='Speedup')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--mtau', type=int)
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
