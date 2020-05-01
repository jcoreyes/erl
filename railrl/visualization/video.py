import os
import os.path as osp
import uuid

from railrl.envs.vae_wrappers import VAEWrappedEnv, ConditionalVAEWrappedEnv

filename = str(uuid.uuid4())

import skvideo.io
import numpy as np
import time

import scipy.misc

from multiworld.core.image_env import ImageEnv
from railrl.core import logger
import pickle


def save_paths(algo, epoch):
    expl_paths = algo.expl_data_collector.get_epoch_paths()
    filename = osp.join(logger.get_snapshot_dir(),
                        'video_{epoch}_vae.p'.format(epoch=epoch))
    pickle.dump(expl_paths, open(filename, "wb"))
    print("saved", filename)
    eval_paths = algo.eval_data_collector.get_epoch_paths()
    filename = osp.join(logger.get_snapshot_dir(),
                        'video_{epoch}_env.p'.format(epoch=epoch))
    pickle.dump(eval_paths, open(filename, "wb"))
    print("saved", filename)


class VideoSaveFunction:
    def __init__(self, env, variant, expl_path_collector=None,
                 eval_path_collector=None):
        self.env = env
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        if 'imsize' not in self.dump_video_kwargs:
            self.dump_video_kwargs['imsize'] = env.imsize
        self.dump_video_kwargs.setdefault("rows", 2)
        # self.dump_video_kwargs.setdefault("columns", 5)
        self.dump_video_kwargs.setdefault("columns", 1)
        self.dump_video_kwargs.setdefault("unnormalize", True)
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 50)
        self.exploration_goal_image_key = self.dump_video_kwargs.pop(
            "exploration_goal_image_key", "decoded_goal_image")
        self.evaluation_goal_image_key = self.dump_video_kwargs.pop(
            "evaluation_goal_image_key", "image_desired_goal")
        self.path_length = variant.get('algo_kwargs', {}).get('max_path_length', 200)
        self.expl_path_collector = expl_path_collector
        self.eval_path_collector = eval_path_collector
        self.variant = variant

    def __call__(self, algo, epoch):
        if self.expl_path_collector:
            expl_paths = self.expl_path_collector.collect_new_paths(
                max_path_length=self.path_length,
                num_steps=self.path_length * 5,
                discard_incomplete_paths=False
            )
        else:
            expl_paths = algo.expl_data_collector.get_epoch_paths()
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(self.logdir,
                                'video_{epoch}_vae.mp4'.format(epoch=epoch))
            dump_paths(self.env,
                       filename,
                       expl_paths,
                       self.exploration_goal_image_key,
                       **self.dump_video_kwargs,
                       )

        if self.eval_path_collector:
            eval_paths = self.eval_path_collector.collect_new_paths(
                max_path_length=self.path_length,
                num_steps=self.path_length * 5,
                discard_incomplete_paths=False
            )
        else:
            eval_paths = algo.eval_data_collector.get_epoch_paths()
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(self.logdir,
                                'video_{epoch}_env.mp4'.format(epoch=epoch))
            dump_paths(self.env,
                       filename,
                       eval_paths,
                       self.evaluation_goal_image_key,
                       **self.dump_video_kwargs,
                       )


class RIGVideoSaveFunction:
    def __init__(self,
        model,
        data_collector,
        tag,
        goal_image_key,
        save_video_period,
        **kwargs
    ):
        self.model = model
        self.data_collector = data_collector
        self.tag = tag
        self.goal_image_key = goal_image_key
        self.dump_video_kwargs = kwargs
        self.save_video_period = save_video_period
        self.logdir = logger.get_snapshot_dir()

    def __call__(self, algo, epoch):
        paths = self.data_collector.get_epoch_paths()
        if epoch % self.save_video_period == 0 or epoch == algo.num_epochs:
            filename = osp.join(self.logdir,
                'video_{epoch}_{tag}.mp4'.format(epoch=epoch, tag=self.tag))
            if self.model:
                for i in range(len(paths)):
                    self.add_decoded_goal_to_path(paths[i])
            dump_paths(None,
                filename,
                paths,
                self.goal_image_key,
                **self.dump_video_kwargs,
            )

    def add_decoded_goal_to_path(self, path):
        latent = path['full_observations'][0]['latent_desired_goal']
        decoded_img = self.model.decode_one_np(latent)
        for i_in_path, d in enumerate(path['full_observations']):
            d[self.goal_image_key] = decoded_img

def add_border(img, border_thickness, border_color):
    imheight, imwidth = img.shape[:2]
    framed_img = np.ones(
        (
            imheight + 2 * border_thickness,
            imwidth + 2 * border_thickness,
            img.shape[2]
        ),
        dtype=np.uint8
    ) * border_color
    framed_img[
        border_thickness:-border_thickness,
        border_thickness:-border_thickness,
        :
    ] = img
    return framed_img


def make_image_fit_into_hwc_format(
        img, output_imwidth, output_imheight, input_image_format
):
    if len(img.shape) == 1:
        if input_image_format == 'HWC':
            hwc_img = img.reshape(output_imheight, output_imwidth, -1)
        elif input_image_format == 'CWH':
            cwh_img = img.reshape(-1, output_imwidth, output_imheight)
            hwc_img = cwh_img.transpose()
        else:
            raise ValueError(input_image_format)
    else:
        a, b, c = img.shape
        # TODO: remove hack
        if a == b and a != c:
            input_image_format = 'HWC'
        elif a != b and b == c:
            input_image_format = 'CWH'
        if input_image_format == 'HWC':
            hwc_img = img
        elif input_image_format == 'CWH':
            hwc_img = img.transpose()
        else:
            raise ValueError(input_image_format)

    if hwc_img.shape == (output_imheight, output_imwidth, 3):
        image_that_fits = hwc_img
    else:
        try:
            import cv2
            image_that_fits = cv2.resize(
                hwc_img,
                dsize=(output_imwidth, output_imheight),
            )
        except ImportError:
            image_that_fits = np.zeros((output_imheight, output_imwidth, 3))
            h, w = hwc_img.shape[:2]
            image_that_fits[:h, :w, :] = hwc_img
    return image_that_fits


def get_image(
        imgs, imwidth, imheight,
        subpad_length=1, subpad_color=255,
        pad_length=1, pad_color=255,
        unnormalize=True,
        image_format='CWH',
):
    hwc_imgs = [
        make_image_fit_into_hwc_format(img, imwidth, imheight, image_format)
        for img in imgs
    ]

    new_imgs = []
    for img in hwc_imgs:
        if unnormalize:
            img = np.uint8(255 * img)
        if subpad_length > 0:
            img = add_border(img, subpad_length, subpad_color)
        new_imgs.append(img)
    final_image = np.concatenate(new_imgs, axis=0)
    if pad_length > 0:
        final_image = add_border(final_image, pad_length, pad_color)
    return final_image


def dump_video(
        env,
        policy,
        filename,
        rollout_function,
        rows=3,
        columns=6,
        pad_length=1,
        pad_color=255,
        subpad_length=1,
        subpad_color=127,
        image_format='HWC',
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        get_extra_imgs=None,
        grayscale=False,
        keys_to_show=None,
):
    """

    :param env:
    :param policy:
    :param filename:
    :param rollout_function:
    :param rows:
    :param columns:
    :param pad_length:
    :param pad_color:
    :param subpad_length:
    :param subpad_color:
    :param do_timer:
    :param horizon:
    :param dirname_to_save_images:
    :param subdirname:
    :param imsize:
    :param get_extra_imgs: A function with type

        def get_extra_imgs(
            path: List[dict],
            index_in_path: int,
            env,
        ) -> List[np.ndarray]:
    :param grayscale:
    :return:
    """
    if get_extra_imgs is None:
        get_extra_imgs = get_generic_env_imgs
    num_channels = 1 if grayscale else 3
    keys_to_show = keys_to_show or ['image_desired_goal', 'image_observation']
    frames = []
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
        )

        l = []
        for i_in_path, d in enumerate(path['full_observations']):
            imgs_to_stack = [d[k] for k in keys_to_show]
            imgs_to_stack += get_extra_imgs(path, i_in_path, env)
            l.append(
                get_image(
                    imgs_to_stack,
                    imwidth=imsize,
                    imheight=imsize,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    subpad_length=subpad_length,
                    subpad_color=subpad_color,
                    image_format=image_format,
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    outputdata = reshape_for_video(frames, N, rows, columns, num_channels)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)


def reshape_for_video(frames, N, rows, columns, num_channels):
    img_height, img_width = frames[0].shape[:2]
    frames = np.array(frames, dtype=np.uint8)
    # TODO: can't we just do path_length = len(frames) / N ?
    path_length = frames.size // (
            N * img_height * img_width * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, img_height, img_width, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, img_height, img_width, num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)
    return outputdata


def get_generic_env_imgs(path, i_in_path, env):
    x_0 = path['full_observations'][0]['image_observation']
    d = path['full_observations'][i_in_path]
    is_vae_env = isinstance(env, VAEWrappedEnv)
    is_conditional_vae_env = isinstance(env, ConditionalVAEWrappedEnv)
    imgs = []
    if is_conditional_vae_env:
        imgs.append(
            np.clip(env._reconstruct_img(d['image_observation'], x_0), 0, 1)
        )
    elif is_vae_env:
        imgs.append(
            np.clip(env._reconstruct_img(d['image_observation']), 0, 1)
        )
    return imgs


def dump_paths(
        env,
        filename,
        paths,
        goal_image_key,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        subpad_length=0,
        subpad_color=127,
        do_timer=True,
        horizon=100,
        dirname_to_save_images=None,
        subdirname="rollouts",
        imsize=84,
        imwidth=None,
        imheight=None,
        num_imgs=3,  # how many vertical images we stack per rollout
        dump_pickle=False,
        unnormalize=True,
        grayscale=False,
        get_extra_imgs=None,
):
    if get_extra_imgs is None:
        get_extra_imgs = get_generic_env_imgs
    # num_channels = env.vae.input_channels
    num_channels = 1 if grayscale else 3
    frames = []

    imwidth = imwidth or imsize  # 500
    imheight = imheight or imsize  # 300
    num_gaps = num_imgs - 1  # 2

    H = num_imgs * imheight  # imsize
    W = imwidth  # imsize

    rows = min(rows, int(len(paths) / columns))
    N = rows * columns
    for i in range(N):
        start = time.time()
        path = paths[i]
        l = []
        for i_in_path, d in enumerate(path['full_observations']):
            imgs = [
                d[goal_image_key],
                d['image_observation'],
            ]
            imgs = imgs + get_extra_imgs(path, i_in_path, env)
            imgs = imgs[:num_imgs]
            l.append(
                get_image(
                    imgs,
                    imwidth,
                    imheight,
                    pad_length=pad_length,
                    pad_color=pad_color,
                    subpad_length=subpad_length,
                    subpad_color=subpad_color,
                    unnormalize=unnormalize,
                )
            )
        frames += l

        if dirname_to_save_images:
            rollout_dir = osp.join(dirname_to_save_images, subdirname, str(i))
            os.makedirs(rollout_dir, exist_ok=True)
            rollout_frames = frames[-101:]
            goal_img = np.flip(rollout_frames[0][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/goal.png", goal_img)
            goal_img = np.flip(rollout_frames[1][:imsize, :imsize, :], 0)
            scipy.misc.imsave(rollout_dir + "/z_goal.png", goal_img)
            for j in range(0, 101, 1):
                img = np.flip(rollout_frames[j][imsize:, :imsize, :], 0)
                scipy.misc.imsave(rollout_dir + "/" + str(j) + ".png", img)
        if do_timer:
            print(i, time.time() - start)

    outputdata = reshape_for_video(frames, N, rows, columns, num_channels)
    skvideo.io.vwrite(filename, outputdata)
    print("Saved video to ", filename)

    print("Pickle?", dump_pickle)
    if dump_pickle:
        pickle_filename = filename[:-4] + ".p"
        pickle.dump(paths, open(pickle_filename, "wb"))


def get_save_video_function(
        rollout_function,
        env,
        policy,
        save_video_period=10,
        imsize=48,
        tag="",
        video_image_env_kwargs=None,
        **dump_video_kwargs
):
    logdir = logger.get_snapshot_dir()

    if not isinstance(env, ImageEnv) and not isinstance(env, VAEWrappedEnv):
        if video_image_env_kwargs is None:
            video_image_env_kwargs = {}
        image_env = ImageEnv(env, imsize, transpose=True, normalize=True,
                             **video_image_env_kwargs)
    else:
        image_env = env
        assert image_env.imsize == imsize, "Imsize must match env imsize"

    def save_video(algo, epoch):
        if epoch % save_video_period == 0 or epoch >= algo.num_epochs - 1:
            filename = osp.join(
                logdir,
                'video_{}_{epoch}_env.mp4'.format(tag, epoch=epoch),
            )
            dump_video(image_env, policy, filename, rollout_function,
                       imsize=imsize, **dump_video_kwargs)
    return save_video