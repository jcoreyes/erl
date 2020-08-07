from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import joblib
import uuid
from rlkit.core import logger

import cv2
import os.path as osp
import os

filename = str(uuid.uuid4())

import skvideo.io
import numpy as np
import time

from rlkit.envs.wrappers import ImageMujocoEnv
import scipy.misc

H = 168
W = 84
PAD = 0 # False
if PAD:
    W += 2 * PAD
    H += 2 * PAD

def open_video_writer(name):
    print(name)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(name, fourcc, 10.0, (84,84), True)

    out = skvideo.io.FFmpegWriter(name, {}, {"-r": "50"})
    return out


def add_border(img):
    img = img.reshape((168, 84, -1))
    img2 = np.ones((H, W, img.shape[2]), dtype=np.uint8) * 255
    img2[PAD:-PAD, PAD:-PAD, :] = img
    return img2


def get_image(goal, obs):
    img = np.concatenate((goal, obs))
    img = np.uint8(255 * img)
    if PAD:
        img = add_border(img)
    return img


def rollout(env, agent, frames, max_path_length=np.inf, animated=False, image_env=None):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param animated:
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    goal = env.get_goal()
    agent.reset()
    # obs, goal = env.goal_obs, env.cur_obs
    # import pdb; pdb.set_trace()
    frames.append(get_image(env.goal_obs, env.cur_obs))
    next_o = None
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        obs = np.hstack((o, goal))
        a, agent_info = agent.get_action(obs)
        next_o, r, d, env_info = env.step(a)
        frames.append(get_image(env.goal_decoded, env.cur_obs))
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def dump_video(
        env,
        policy,
        filename,
        ROWS=3,
        COLUMNS=6,
        do_timer=True,
        horizon=100,
        image_env=None,
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
        paths.append(rollout(
            env,
            policy,
            frames,
            max_path_length=horizon,
            animated=False,
            image_env=image_env,
        ))
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

    return paths

from rlkit.images.camera import sawyer_init_camera, \
    sawyer_init_camera_zoomed_in

def simulate_policy(args):
    data = joblib.load(args.file)
    if 'eval_policy' in data:
        policy = data['eval_policy']
    elif 'policy' in data:
        policy = data['policy']
    elif 'exploration_policy' in data:
        policy = data['exploration_policy']
    elif 'naf_policy' in data:
        policy = data['naf_policy']
    elif 'optimizable_qfunction' in data:
        qf = data['optimizable_qfunction']
        policy = qf.implicit_policy
    else:
        raise Exception("No policy found in loaded dict. Keys: {}".format(
            data.keys()
        ))

    env = data['env']
    if isinstance(env, RemoteRolloutEnv):
        env = env._wrapped_env
    print("Policy loaded")

    env.mode("video_env")
    env.decode_goals = True

    image_env = ImageMujocoEnv(
        env._wrapped_env._wrapped_env,
        84,
        init_camera=None,
        camera_name="topview",
        transpose=True,
        normalize=True,
    )
    # env.image_env = image_env

    if args.enable_render:
        # some environments need to be reconfigured for visualization
        env.enable_render()

    if args.gpu:
        set_gpu_mode(True)
        policy.to(ptu.device)
        if hasattr(env, "vae"):
            env.vae.to(ptu.device)
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
        ROWS=ROWS, COLUMNS=COLUMNS, horizon=args.H, image_env=image_env,
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
    parser.add_argument('--pause', action='store_true')
    parser.add_argument('--hide', action='store_true')
    parser.add_argument('--enable_render', action='store_true')
    args = parser.parse_args()

    simulate_policy(args)
