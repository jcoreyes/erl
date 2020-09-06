import gym
import numpy as np

from rlkit.envs.pygame import pnp_util
from rlkit.torch.sets import set_creation
from multiworld.envs.pygame import PickAndPlaceEnv
from rlkit.envs.images import EnvRenderer
from multiworld import register_all_envs



def main():
    register_all_envs()
    # env = PickAndPlaceEnv(
    #     # Environment dynamics
    #     action_scale=1.0,
    #     boundary_dist=4,
    #     ball_radius=1.5,
    #     object_radius=1.,
    #     ball_visual_radius=1.5,
    #     object_visual_radius=1.,
    #     min_grab_distance=1.,
    #     walls=None,
    #     # Rewards
    #     action_l2norm_penalty=0,
    #     reward_type="dense",
    #     success_threshold=0.60,
    #     # Reset settings
    #     fixed_goal=None,
    #     # Visualization settings
    #     images_are_rgb=True,
    #     render_dt_msec=0,
    #     render_onscreen=False,
    #     render_size=84,
    #     show_goal=False,
    #     goal_samplers=None,
    #     goal_sampling_mode='random',
    #     num_presampled_goals=10000,
    #     object_reward_only=False,
    #
    #     init_position_strategy='random',
    #     num_objects=1,
    # )
    env = gym.make('OneObject-PickAndPlace-BigBall-RandomInit-2D-v1')

    renderer = EnvRenderer(
        output_image_format='CHW',
        width=28,
        height=28,
    )
    import cv2
    from PIL import Image
    n = 12800
    imgs = []
    for _ in range(n):
        env.reset()
        img = renderer(env)
        # cv2.imshow('img', img.transpose())
        # cv2.waitKey(100)
        imgs.append(img)
    imgs = np.array(imgs)
    np.save(
        '/home/vitchyr/mnt/log/manual-upload/sets/OneObject-PickAndPlace-BigBall-RandomInit-2D-v1-ungrouped-train-28x28.npy',
        imgs,
    )
    # for set in sets:
    # set_creation.save(
    #     sets,
    #     'manual-upload/sets/hand2xy_hand2x_1obj2xy_1obj2x_num_objs_1.pickle',
    # )


if __name__ == '__main__':
    main()