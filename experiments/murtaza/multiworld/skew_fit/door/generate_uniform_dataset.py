import os.path as osp
#import cv2
import numpy as np

from multiworld.core.image_env import unormalize_image, ImageEnv
from rlkit.misc.asset_loader import load_local_or_remote_file
import rlkit.torch.pytorch_util as ptu

def generate_uniform_dataset_door(
        num_imgs=1000,
        use_cached_dataset=False,
        init_camera=None,
        imsize=48,
        policy_file=None,
        show=False,
        path_length=100,
        save_file_prefix=None,
        env_id=None,
        tag='',
        dataset_path=None
):
    if dataset_path is not None:
        dataset = load_local_or_remote_file(dataset_path)
        return dataset
    import gym
    from gym.envs import registration
    # trigger registration
    import multiworld.envs.pygame
    import multiworld.envs.mujoco
    env = gym.make(env_id)
    env = ImageEnv(
        env,
        imsize,
        init_camera=init_camera,
        transpose=True,
        normalize=True,
    )
    env.non_presampled_goal_img_is_garbage = True
    if save_file_prefix is None and env_id is not None:
        save_file_prefix = env_id
    filename = "/tmp/{}_N{}_imsize{}uniform_images_{}.npy".format(
        save_file_prefix,
        str(num_imgs),
        env.imsize,
        tag,
    )
    if use_cached_dataset and osp.isfile(filename):
        images = np.load(filename)
        print("Loaded data from {}".format(filename))
        return images

    policy_file = load_local_or_remote_file(policy_file)
    policy = policy_file['policy']
    policy.to(ptu.device)
    print('Sampling Uniform Dataset')
    dataset = np.zeros((num_imgs, 3 * env.imsize ** 2), dtype=np.uint8)
    for j in range(num_imgs):
        obs = env.reset()
        policy.reset()
        for i in range(path_length):
            policy_obs = np.hstack((
                obs['state_observation'],
                obs['state_desired_goal'],
            ))
            action, _ = policy.get_action(policy_obs)
            obs, _, _, _ = env.step(action)
        img_f = obs['image_observation']
        if show:
            img = obs['image_observation']
            img = img.reshape(3, env.imsize, env.imsize).transpose()
            img = img[::-1, :, ::-1]
            cv2.imshow('img', img)
            cv2.waitKey(1)
        print(j)
        dataset[j, :] = unormalize_image(img_f)
    temp = env.reset_free
    env.reset_free = True
    env.reset()
    env.reset_free = temp
    np.save(filename, dataset)
    print("Saving file to {}".format(filename))
    return dataset