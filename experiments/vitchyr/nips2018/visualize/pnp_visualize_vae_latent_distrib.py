from rlkit.envs.remote import RemoteRolloutEnv
from rlkit.samplers.util import rollout
from rlkit.torch.core import PyTorchModule
import matplotlib.pyplot as plt
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import pickle
import uuid
from rlkit.core import logger
from torchvision.utils import save_image
import numpy as np
import rlkit.torch.pytorch_util as ptu
import cv2


def get_info(goals):
    imgs = []
    latent_mus = []
    latent_sigmas = []
    for goal in goals:
        env.set_to_goal({
            'state_desired_goal': goal,
        })
        flat_img = env._get_flat_img()
        img = flat_img.reshape(
            3,
            84,
            84,
        )
        mu, sigma = vae.encode(ptu.np_to_var(flat_img))
        latent_mus.append(ptu.get_numpy(mu))
        latent_sigmas.append(ptu.get_numpy(sigma))
        imgs.append(img)
    return imgs, latent_mus, latent_sigmas


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    args = parser.parse_args()
    vae = pickle.load(open(args.dir + "/vae.pkl", "rb"))
    data = pickle.load(open(args.dir + "/params.pkl", "rb"))
    env = data['env']
    env = env.wrapped_env
    env.reset()

    hand_xyz = env.get_endeff_pos()
    obj_xyz = env.get_obj_pos()

    #  ------------------ X axis - obj
    goals = []
    for x in np.arange(
            env.hand_and_obj_space.low[3],
            env.hand_and_obj_space.high[3],
            0.01,
    ):
        new_obj_xyz = obj_xyz.copy()
        new_obj_xyz[0] = x
        goals.append(
            np.hstack((hand_xyz, new_obj_xyz))
        )
    imgs, latent_mus, latent_sigmas = get_info(goals)

    mu_stds = np.std(np.vstack(latent_mus), axis=0)
    plt.bar(np.arange(len(mu_stds)), mu_stds)
    plt.title("X-axis puck sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Mean std")
    plt.show()

    sigma_stds = np.mean(np.vstack(latent_sigmas), axis=0)
    plt.bar(np.arange(len(sigma_stds)), sigma_stds)
    plt.title("X-axis puck sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Sigma std")
    plt.show()

    imgs = np.array(imgs)
    imgs = ptu.FloatTensor(imgs)
    save_image(imgs, 'x-puck-sweep.png')

    #  ------------------ X axis - arm
    goals = []
    for x in np.arange(env.hand_low[0], env.hand_high[0], 0.01):
        new_hand_xyz = hand_xyz.copy()
        new_hand_xyz[0] = x
        goals.append(
            np.hstack((new_hand_xyz, obj_xyz))
        )
    imgs, latent_mus, latent_sigmas = get_info(goals)

    mu_stds = np.std(np.vstack(latent_mus), axis=0)
    plt.bar(np.arange(len(mu_stds)), mu_stds)
    plt.title("X-axis arm sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Mean std")
    plt.show()

    sigma_stds = np.mean(np.vstack(latent_sigmas), axis=0)
    plt.bar(np.arange(len(sigma_stds)), sigma_stds)
    plt.title("X-axis arm sweep")
    plt.xlabel("latent dim")
    plt.ylabel("Sigma std")
    plt.show()

    imgs = np.array(imgs)
    imgs = ptu.FloatTensor(imgs)
    img = save_image(imgs, 'x-arm-sweep.png')
