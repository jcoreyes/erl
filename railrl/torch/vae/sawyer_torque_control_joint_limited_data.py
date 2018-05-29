import time

import numpy as np
import os.path as osp

from railrl.envs.mujoco.sawyer_reach_torque_env import SawyerReachTorqueEnv
from railrl.envs.wrappers import ImageMujocoEnv
from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.images.camera import sawyer_torque_env_camera
import cv2

from railrl.misc.asset_loader import local_path_from_s3_or_local_path
from railrl.policies.simple import ZeroPolicy, RandomPolicy


def generate_vae_dataset(
        N=10000, num_divisions=10, test_p=0.9, use_cached=True, imsize=84, show=True,
        dataset_path=None, save_state_info=False,
):
    img_file = "/home/murtaza/vae_data/sawyer_torque_control_images" + str(N)
    state_file = "/home/murtaza/vae_data/sawyer_torque_control_states" + str(N)
    info = {}

    if dataset_path is not None:
        img_file = local_path_from_s3_or_local_path(dataset_path)
        dataset = np.load(img_file)
    elif use_cached and osp.isfile(img_file):
        dataset = np.load(img_file)
        print("loaded data from saved file", img_file)
    else:
        now = time.time()
        env = SawyerReachTorqueEnv(keep_vel_in_obs=False, hide_goal=True)
        obs_dim = env.observation_space.low.size
        env = ImageMujocoEnv(
            env, imsize,
            transpose=True,
            init_camera=sawyer_torque_env_camera,
            normalize=True,
        )
        info['env'] = env
        # policy = ZeroPolicy(env.action_space.low.size)
        policy = RandomPolicy(env.action_space)
        es = OUStrategy(action_space=env.action_space, theta=0)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy,
        )
        single_set_size = int(N/num_divisions)
        dataset = np.zeros((single_set_size, imsize * imsize * 3))
        states = np.zeros((single_set_size, obs_dim))
        count = 1
        for i in range(N):
            if i % single_set_size==0 and i!=0:
                np.save(img_file + '_'+str(count)+'.npy', dataset)
                np.save(state_file+'_'+str(count)+'.npy', states)
                dataset = np.zeros((single_set_size, imsize * imsize * 3))
                states = np.zeros((single_set_size, obs_dim))
                count+=1
            # Move the goal out of the image
            env.wrapped_env.set_goal(np.array([100, 100, 100]))
            if i %50==0:
                print('Reset')
                env.reset()
                exploration_policy.reset()
            for _ in range(75):
                action = exploration_policy.get_action()[0]*10
                env.wrapped_env.step(
                    action
                )
            img = env.step(env.action_space.sample())[0]
            states[i%single_set_size,:] = env._wrapped_env._get_obs()
            dataset[i%single_set_size, :] = img
            if show:
                cv2.imshow('img', img.reshape(3, 84, 84).transpose())
                cv2.waitKey(1)
            print(i)


        print("done making training data", time.time() - now)
        np.save(img_file + '_' + str(count) + '.npy', dataset)
        np.save(state_file + '_' + str(count) + '.npy', states)
    n = int(N * test_p)
    train_dataset = dataset[:n, :]
    test_dataset = dataset[n:, :]
    return train_dataset, test_dataset, info


if __name__ == "__main__":
    generate_vae_dataset(10000, num_divisions=10, use_cached=False, show=True, save_state_info=True)
