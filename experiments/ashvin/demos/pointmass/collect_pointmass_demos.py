from rlkit.demos.collect_demo import collect_demos, SpaceMouseExpert
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

from multiworld.envs.pygame.point2d import Point2DWallEnv

import numpy as np

if __name__ == '__main__':
    expert = SpaceMouseExpert(xyz_dims=2)

    env = Point2DWallEnv(
        render_onscreen=False,
        images_are_rgb=True,
    )
    env = ImageEnv(env,
        non_presampled_goal_img_is_garbage=True,
        recompute_reward=False,
        # transpose=True,
        # init_camera=sawyer_pusher_camera_upright_v2,
    )

    collect_demos(env, expert, "pointmass_demos_100.npy", 100)
