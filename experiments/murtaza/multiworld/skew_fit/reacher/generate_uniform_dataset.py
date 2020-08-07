import os.path as osp
#import cv2
import numpy as np
from multiworld.core.image_env import unormalize_image, ImageEnv
from rlkit.misc.asset_loader import load_local_or_remote_file

def generate_uniform_dataset_reacher(
        env_class=None,
        env_kwargs=None,
        num_imgs=1000,
        use_cached_dataset=False,
        init_camera=None,
        imsize=48,
        show=False,
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
    if not env_class or not env_kwargs:
        env = gym.make(env_id)
    else:
        env = env_class(**env_kwargs)
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

    print('Sampling Uniform Dataset')
    dataset = np.zeros((num_imgs, 3 * env.imsize ** 2), dtype=np.uint8)
    for j in range(num_imgs):
        obs = env.reset()
        env.set_to_goal(env.get_goal())
        img_f = env._get_flat_img()
        if show:
            img = img_f.reshape(3, env.imsize, env.imsize).transpose()
            img = img[::-1, :, ::-1]
            cv2.imshow('img', img)
            cv2.waitKey(1)
        print(j)
        dataset[j, :] = unormalize_image(img_f)
    np.save(filename, dataset)
    print("Saving file to {}".format(filename))
    return dataset
