from rlkit.demos.collect_demo import collect_demos, SpaceMouseExpert
from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multienv import SawyerPushAndReachXYEasyEnv
from multiworld.envs.pygame.point2d import Point2DWallEnv

import numpy as np

if __name__ == '__main__':
    expert = SpaceMouseExpert(
        xyz_dims=2,
        xyz_remap=[1, 0, 2],
        xyz_scale=[-1, -1, -1],
    )

    env = SawyerPushAndReachXYEasyEnv(
        hide_goal=True,
        reward_info=dict(
            type="state_distance",
        ),
    )
    env = ImageEnv(env,
        recompute_reward=False,
        # transpose=True,
        init_camera=sawyer_pusher_camera_upright_v2,
    )

    collect_demos(env, expert, "pusher_demos_100.npy", 100)
