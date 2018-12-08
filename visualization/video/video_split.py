import skvideo.io
import skvideo.datasets
import numpy as np
import cv2

def left_justify_video(video):
    """
    Sometimes the videos have a variable amount of black on the left. Remove
    all that balack.
    """
    new_video = np.zeros_like(video)
    for t, frame in enumerate(video):
        is_black = (frame < 10)
        column_is_black = is_black.all(axis=(0, 2))
        first_colored_column_i = 0
        while column_is_black[first_colored_column_i]:
            first_colored_column_i += 1
        last_colored_column_i = column_is_black.size - 1
        while column_is_black[last_colored_column_i]:
            last_colored_column_i -= 1
        width = last_colored_column_i - first_colored_column_i + 1
        new_video[t, :, :width, :] = (
            frame[:, first_colored_column_i:last_colored_column_i + 1, :]
        )

    return new_video


def remove_black_from_right(video):
    is_black = (video < 30)
    column_is_black = is_black.all(axis=(0, 1, 3))
    last_colored_column_i = column_is_black.size - 1
    while column_is_black[last_colored_column_i]:
        last_colored_column_i -= 1

    if last_colored_column_i % 2 == 1:
        last_colored_column_i -= 1

    return video[:, :, :last_colored_column_i, :]


def replace_black_with_white(video):
    is_black = (video < 30).all(axis=3)
    video[is_black, :] = 255
    return video


def split_video_top_bottom(video):
    n_rows = video.shape[1] // 2
    return video[:, :n_rows, :, :], video[:, n_rows:, :, :]


def add_white_mid_bar(video):
    video = video.copy()
    mid = video.shape[1] // 2
    video[:, mid-3:mid+3, :, :] = 255
    return video


def swap_top_bottom(video):
    video = video.copy()
    mid = video.shape[1] // 2
    new_video = np.zeros_like(video)
    new_video[:, mid:, ...] = video[:, :mid, ...]
    new_video[:, :mid, ...] = video[:, mid:, ...]
    return new_video


if __name__ == "__main__":  # how to use this:
    output_dir = "/home/vitchyr/Dropbox/Berkeley/Research/nips2018/talk/generated-videos/"


    input_file = "/home/vitchyr/Videos/research/rig-blog-post/door.gif"
    output_name = "door"

    # input_file = (
        # "/home/vitchyr/Dropbox/Berkeley/Research/nips2018/talk/pusher2_env.mp4"
    # )
    # output_name = "pusher2-env"

    input_file = (
        "/home/vitchyr/Dropbox/Berkeley/Research/nips2018/talk/pusher2_vae.mp4"
    )
    output_name = "pusher2-vae"

    # input_file = (
        # "/home/vitchyr/Videos/research/rig-blog-post/pick-and-place.gif"
    # )
    # output_name = "pnp"

    input_file = (
        "/home/vitchyr/Videos/research/rig-blog-post/real_reaching_vae.mp4"
    )
    output_name = "real_reacher_vae"

    # input_file = (
        # "/home/vitchyr/Videos/research/rig-blog-post/real_reaching_env.mp4"
    # )
    # output_name = "real_reacher_env"

    # input_file = (
        # "/home/vitchyr/Videos/research/rig-blog-post/real_world_pushing.mp4"
    # )
    # output_name = "real_pusher"

    video = skvideo.io.vread(input_file)
    new_video = left_justify_video(video)
    new_video = remove_black_from_right(new_video)
    # new_video = replace_black_with_white(new_video)
    new_video = swap_top_bottom(new_video)
    new_video = add_white_mid_bar(new_video)
    top, bottom = split_video_top_bottom(new_video)
    skvideo.io.vwrite(
        output_dir + output_name + "-full.mp4",
        new_video,
        outputdict={
            '-pix_fmt': 'yuv420p',  # for the videos to play
        },
    )
    skvideo.io.vwrite(
        output_dir + output_name + "-white-mid.mp4",
        add_white_mid_bar(new_video),
        outputdict={
            '-pix_fmt': 'yuv420p',  # for the videos to play
        },
    )
    skvideo.io.vwrite(
        output_dir + output_name + "-top.mp4",
        top,
        outputdict={
            '-pix_fmt': 'yuv420p',  # for the videos to play
        },
    )
    skvideo.io.vwrite(
        output_dir + output_name + "-bottom.mp4",
        bottom,
        outputdict={
            '-pix_fmt': 'yuv420p',
        },
    )
    cv2.imwrite(
        output_dir + output_name + "-top-first-frame.png", top[0][..., ::-1]
    )
    cv2.imwrite(
        output_dir + output_name + "-bottom-first-frame.png", bottom[0][..., ::-1]
    )
