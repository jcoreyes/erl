from rlkit.demos.collect_demo import collect_demos, SpaceMouseExpert
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_init_camera_zoomed_in

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerMultiobjectEnv
from multiworld.envs.pygame.point2d import Point2DWallEnv

import numpy as np

if __name__ == '__main__':
    expert = SpaceMouseExpert(
        xyz_dims=2,
        xyz_remap=[1, 0, 2],
        xyz_scale=[-1, -1, -1],
    )

    x_low = -0.2
    x_high = 0.2
    y_low = 0.5
    y_high = 0.7
    t = 0.03
    env = SawyerMultiobjectEnv(
        num_objects=1,
        reset_to_initial_position=False,
        puck_goal_low=(x_low + t + t, y_low + t),
        puck_goal_high=(x_high - t - t, y_high - t),
        hand_goal_low=(x_low, y_low),
        hand_goal_high=(x_high, y_high),
        mocap_low=(x_low, y_low, 0.0),
        mocap_high=(x_high, y_high, 0.5),
        object_low=(x_low + t + t, y_low + t, 0.0),
        object_high=(x_high - t - t, y_high - t, 0.5),
        preload_obj_dict=[
            dict(color2=(0.1, 0.1, 0.9)),
        ],
    )
    env = ImageEnv(env,
        recompute_reward=False,
        # transpose=True,
        init_camera=sawyer_init_camera_zoomed_in,
    )

    collect_demos(env, expert, "pusher_reset_free_demos_100b.npy", 100)
