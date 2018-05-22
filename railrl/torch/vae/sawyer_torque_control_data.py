import time

import numpy as np
import os.path as osp

from railrl.envs.mujoco.sawyer_reach_torque_env import SawyerReachTorqueEnv
from railrl.envs.wrappers import ImageMujocoEnv
from railrl.images.camera import sawyer_torque_env_camera
import cv2

from railrl.misc.asset_loader import local_path_from_s3_or_local_path


def generate_vae_dataset(
        N=10000, test_p=0.9, use_cached=True, imsize=84, show=True,
        dataset_path=None, save_state_info=False,
):
    img_file = "/tmp/sawyer_torque_control_imgs" + str(N) + ".npy"
    state_file = "/tmp/sawyer_torque_control_states" + str(N) + ".npy"
    info = {}
    if dataset_path is not None:
        img_file = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(img_file)
    elif use_cached and osp.isfile(img_file):
        dataset = np.load(img_file)
        print("loaded data from saved file", img_file)
    else:
        now = time.time()
        env = SawyerReachTorqueEnv(keep_vel_in_obs=False)
        obs_dim = env.observation_space.low.size
        env = ImageMujocoEnv(
            env, imsize,
            transpose=True,
            init_camera=sawyer_torque_env_camera,
            normalize=True,
        )
        info['env'] = env

        dataset = np.zeros((N, imsize * imsize * 3))
        states = np.zeros((N, obs_dim))
        for i in range(N):
            # Move the goal out of the image
            env.wrapped_env.set_goal(np.array([100, 100, 100]))
            env.reset()
            action = env.wrapped_env.action_space.sample()
            for _ in range(100):
                env.wrapped_env.step(
                    action
                )
            img = env.step(env.action_space.sample())[0]
            states[i,:] = env._wrapped_env._get_obs()
            dataset[i, :] = img
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)
        print("done making training data", img_file, time.time() - now)
        np.save(img_file, dataset)
        np.save(state_file, states)

    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(5000, use_cached=False, show=True)
